from __future__ import annotations

import hashlib
import hmac
import html
import re
import secrets
import sqlite3
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

app = FastAPI(title='Loan Risk Prediction App')
USER_DATABASE = Path(__file__).with_name('users.db')

AUTH_USERS = {
    'admin': 'admin123',
    'analyst': 'loan123',
}

PURPOSES = [
    'all_other',
    'credit_card',
    'debt_consolidation',
    'educational',
    'home_improvement',
    'major_purchase',
    'small_business',
]
MODEL_NAME = 'Random Forest Classifier'
MODEL_TREES = 600


def initialize_user_store() -> str:
    with sqlite3.connect(USER_DATABASE) as connection:
        connection.execute(
            '''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY COLLATE NOCASE,
                full_name TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL
            )
            '''
        )
        connection.execute(
            '''
            CREATE TABLE IF NOT EXISTS app_settings (
                setting_name TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL
            )
            '''
        )
        secret = connection.execute(
            'SELECT setting_value FROM app_settings WHERE setting_name = ?',
            ('session_secret',),
        ).fetchone()
        if secret is None:
            secret_value = secrets.token_hex(32)
            connection.execute(
                'INSERT INTO app_settings (setting_name, setting_value) VALUES (?, ?)',
                ('session_secret', secret_value),
            )
            return secret_value
        return str(secret[0])


SESSION_SECRET = bytes.fromhex(initialize_user_store())


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 310_000)
    return salt.hex(), password_hash.hex()


def authenticate_user(username: str, password: str) -> bool:
    normalized_username = username.strip().lower()
    demo_password = AUTH_USERS.get(normalized_username)
    if demo_password is not None:
        return hmac.compare_digest(demo_password, password)

    with sqlite3.connect(USER_DATABASE) as connection:
        user = connection.execute(
            'SELECT password_salt, password_hash FROM users WHERE username = ?',
            (normalized_username,),
        ).fetchone()

    if user is None:
        return False

    salt = bytes.fromhex(user[0])
    _, calculated_hash = hash_password(password, salt)
    return hmac.compare_digest(user[1], calculated_hash)


def load_dataset() -> pd.DataFrame:
    return pd.read_csv('loan_data.csv')


def train_model():
    loans = load_dataset()
    final_data = pd.get_dummies(loans, columns=['purpose'], drop_first=True)
    X = final_data.drop('not.fully.paid', axis=1)
    y = final_data['not.fully.paid']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=101, stratify=y
    )
    model = RandomForestClassifier(n_estimators=MODEL_TREES, random_state=101)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    risk_probabilities = model.predict_proba(X_test)[:, 1]
    evaluation = {
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'f1': float(f1_score(y_test, preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, risk_probabilities)),
        'confusion_matrix': confusion_matrix(y_test, preds, labels=[0, 1]).tolist(),
        'test_size': len(y_test),
    }
    return model, round(float(accuracy), 4), X.columns.tolist(), evaluation


MODEL, MODEL_ACCURACY, MODEL_COLUMNS, MODEL_EVALUATION = train_model()


def build_single_record(payload: dict[str, object]) -> pd.DataFrame:
    purpose = str(payload.get('purpose', ''))
    if purpose not in PURPOSES:
        raise ValueError(f'Invalid purpose. Valid values: {PURPOSES}')

    row = {column: 0 for column in MODEL_COLUMNS}
    row['credit.policy'] = int(payload['credit_policy'])
    row['int.rate'] = float(payload['int_rate']) / 100.0
    row['installment'] = float(payload['installment'])
    row['log.annual.inc'] = float(payload['log_annual_inc'])
    row['dti'] = float(payload['dti'])
    row['fico'] = int(payload['fico'])
    row['days.with.cr.line'] = float(payload['days_with_cr_line'])
    row['revol.bal'] = int(payload['revol_bal'])
    row['revol.util'] = float(payload['revol_util'])
    row['inq.last.6mths'] = int(payload['inq_last_6mths'])
    row['delinq.2yrs'] = int(payload['delinq_2yrs'])
    row['pub.rec'] = int(payload['pub_rec'])

    purpose_col = f'purpose_{purpose}'
    if purpose_col in row:
        row[purpose_col] = 1

    return pd.DataFrame([row], columns=MODEL_COLUMNS)


def predict_payload(payload: dict[str, object]) -> dict[str, object]:
    input_df = build_single_record(payload)
    prediction = int(MODEL.predict(input_df)[0])
    probabilities = MODEL.predict_proba(input_df)[0]
    risk_probability = float(probabilities[1])
    safe_probability = float(probabilities[0])
    label = 'Likely not fully paid' if prediction == 1 else 'Likely fully paid'
    return {
        'prediction': prediction,
        'label': label,
        'probability_not_fully_paid': risk_probability,
        'probability_fully_paid': safe_probability,
        'model_accuracy': MODEL_ACCURACY,
    }


def set_session_cookie(response: Response, username: str) -> None:
    signature = hmac.new(SESSION_SECRET, username.encode('utf-8'), hashlib.sha256).hexdigest()
    response.set_cookie(
        key='session_user',
        value=f'{username}.{signature}',
        httponly=True,
        samesite='lax',
        max_age=3600,
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key='session_user')


def get_logged_in_user(request: Request) -> str | None:
    cookie = request.cookies.get('session_user', '')
    username, separator, signature = cookie.rpartition('.')
    if not separator or not username:
        return None
    expected_signature = hmac.new(SESSION_SECRET, username.encode('utf-8'), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        return None
    if username in AUTH_USERS:
        return username
    with sqlite3.connect(USER_DATABASE) as connection:
        user = connection.execute('SELECT 1 FROM users WHERE username = ?', (username,)).fetchone()
    return username if user else None


def render_login_page(message: str = '') -> HTMLResponse:
    error_html = f'<p class="message error">{html.escape(message)}</p>' if message else ''
    return HTMLResponse(
        f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>Login | Loan Risk Prediction</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: linear-gradient(135deg, #eef4ff, #e0f2fe); margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }}
                .card {{ width: min(420px, 90vw); background: white; border-radius: 18px; box-shadow: 0 12px 32px rgba(15, 23, 42, 0.12); padding: 28px; }}
                h1 {{ margin-top: 0; text-align: center; color: #0f172a; }}
                form {{ display: grid; gap: 14px; }}
                label {{ font-weight: 600; color: #334155; }}
                input {{ padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 15px; }}
                button {{ padding: 12px; border: none; border-radius: 10px; background: #2563eb; color: white; font-size: 16px; font-weight: 700; cursor: pointer; }}
                .message {{ padding: 10px 12px; border-radius: 8px; margin-bottom: 16px; text-align: center; }}
                .error {{ background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}
                .demo {{ margin-top: 16px; font-size: 14px; color: #475569; text-align: center; }}
                .demo strong {{ color: #0f172a; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Login</h1>
                {error_html}
                <form method="post" action="/login">
                    <div>
                        <label for="username">Username</label>
                        <input id="username" name="username" type="text" placeholder="Enter username" required />
                    </div>
                    <div>
                        <label for="password">Password</label>
                        <input id="password" name="password" type="password" placeholder="Enter password" required />
                    </div>
                    <button type="submit">Sign In</button>
                </form>
                <div class="demo"><strong>Demo credentials:</strong> admin / admin123 or analyst / loan123</div>
                <div class="demo">New here? <a href="/register">Create an account</a></div>
            </div>
        </body>
        </html>
        '''
    )


def render_register_page(message: str = '') -> HTMLResponse:
    error_html = f'<p class="message error">{html.escape(message)}</p>' if message else ''
    return HTMLResponse(
        f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>Register | Loan Risk Prediction</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: linear-gradient(135deg, #eef4ff, #e0f2fe); margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }}
                .card {{ width: min(420px, 90vw); background: white; border-radius: 18px; box-shadow: 0 12px 32px rgba(15, 23, 42, 0.12); padding: 28px; }}
                h1 {{ margin-top: 0; text-align: center; color: #0f172a; }}
                form {{ display: grid; gap: 14px; }}
                label {{ font-weight: 600; color: #334155; }}
                input {{ box-sizing: border-box; width: 100%; padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 15px; }}
                button {{ padding: 12px; border: none; border-radius: 10px; background: #2563eb; color: white; font-size: 16px; font-weight: 700; cursor: pointer; }}
                .message {{ padding: 10px 12px; border-radius: 8px; margin-bottom: 16px; text-align: center; }}
                .error {{ background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}
                .help {{ margin: 14px 0 0; color: #64748b; font-size: 13px; }}
                .demo {{ margin-top: 16px; font-size: 14px; color: #475569; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Create account</h1>
                {error_html}
                <form method="post" action="/register">
                    <div>
                        <label for="full_name">Full name</label>
                        <input id="full_name" name="full_name" type="text" maxlength="100" autocomplete="name" required />
                    </div>
                    <div>
                        <label for="username">Username</label>
                        <input id="username" name="username" type="text" minlength="3" maxlength="32" pattern="[A-Za-z0-9._\\-]+" autocomplete="username" required />
                    </div>
                    <div>
                        <label for="password">Password</label>
                        <input id="password" name="password" type="password" minlength="8" autocomplete="new-password" required />
                    </div>
                    <div>
                        <label for="confirm_password">Confirm password</label>
                        <input id="confirm_password" name="confirm_password" type="password" minlength="8" autocomplete="new-password" required />
                    </div>
                    <button type="submit">Register</button>
                </form>
                <p class="help">Username: 3–32 letters, numbers, dots, underscores, or hyphens. Password: at least 8 characters.</p>
                <div class="demo">Already registered? <a href="/login">Sign in</a></div>
            </div>
        </body>
        </html>
        '''
    )


def render_dashboard_page(username: str) -> HTMLResponse:
    loans = load_dataset()
    encoded = pd.get_dummies(loans, columns=['purpose'], drop_first=True)
    X = encoded.drop('not.fully.paid', axis=1)
    y = loans['not.fully.paid']
    model_predictions = MODEL.predict(X)
    model_probabilities = MODEL.predict_proba(X)[:, 1]

    total_loans = len(loans)
    actual_default = int((y == 1).sum())
    predicted_default = int((model_predictions == 1).sum())
    predicted_paid = int((model_predictions == 0).sum())
    avg_risk = float(model_probabilities.mean())
    safe_share = predicted_paid / total_loans * 100 if total_loans else 0
    risk_share = predicted_default / total_loans * 100 if total_loans else 0

    purpose_risk = (
        pd.DataFrame({
            'purpose': loans['purpose'],
            'risk_probability': model_probabilities,
        })
        .groupby('purpose', as_index=False)['risk_probability']
        .mean()
        .sort_values('risk_probability', ascending=False)
    )

    rows = []
    for index, record in loans.head(12).iterrows():
        row_pred = int(model_predictions[index])
        risk = float(model_probabilities[index])
        rows.append(
            f'''<tr>
                <td>{index + 1}</td>
                <td>{record['purpose']}</td>
                <td>{record['fico']}</td>
                <td>{record['int.rate']}</td>
                <td>{record['dti']}</td>
                <td>{'Default' if record['not.fully.paid'] == 1 else 'Fully paid'}</td>
                <td>{'Risk' if row_pred == 1 else 'Safe'}</td>
                <td>{risk:.2%}</td>
            </tr>'''
        )

    purpose_rows = ''.join(
        f'''<tr><td>{row['purpose']}</td><td>{row['risk_probability']:.2%}</td></tr>'''
        for _, row in purpose_risk.iterrows()
    )
    purpose_bars = ''.join(
        f'''<div class="bar-row">
            <span>{html.escape(str(row['purpose']).replace('_', ' ').title())}</span>
            <div class="bar-track" role="img" aria-label="{html.escape(str(row['purpose']))}: {row['risk_probability']:.1%} estimated risk">
                <div class="bar purpose-bar" style="width:{row['risk_probability'] * 100:.2f}%"></div>
            </div>
            <strong>{row['risk_probability']:.1%}</strong>
        </div>'''
        for _, row in purpose_risk.iterrows()
    )

    metric_bars = ''.join(
        f'''<div class="metric-row">
            <span>{label}</span>
            <div class="bar-track" role="img" aria-label="{label}: {score:.1%}">
                <div class="bar metric-bar" style="width:{score * 100:.2f}%"></div>
            </div>
            <strong>{score:.1%}</strong>
        </div>'''
        for label, score in (
            ('Accuracy', MODEL_ACCURACY),
            ('Precision', MODEL_EVALUATION['precision']),
            ('Recall', MODEL_EVALUATION['recall']),
            ('F1 score', MODEL_EVALUATION['f1']),
            ('ROC-AUC', MODEL_EVALUATION['roc_auc']),
        )
    )
    true_paid, false_risk = MODEL_EVALUATION['confusion_matrix'][0]
    missed_risk, detected_risk = MODEL_EVALUATION['confusion_matrix'][1]

    return HTMLResponse(
        f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>Dashboard | Loan Risk Prediction</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #f8fafc; margin: 0; color: #0f172a; }}
                .header {{ display: flex; justify-content: space-between; align-items: center; padding: 18px 32px; background: #0f172a; color: white; }}
                nav {{ display: flex; gap: 16px; align-items: center; }}
                nav a {{ color: white; text-decoration: none; font-weight: 600; }}
                .main {{ max-width: 1200px; margin: 28px auto; padding: 0 18px 40px; }}
                .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 18px; margin-bottom: 28px; }}
                .card {{ background: white; border-radius: 16px; padding: 18px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08); }}
                .card h3 {{ margin: 0 0 10px; color: #475569; font-size: 14px; text-transform: uppercase; letter-spacing: 0.08em; }}
                .card .value {{ font-size: 2rem; font-weight: 700; margin: 0; }}
                .card .sub {{ margin-top: 6px; font-size: 13px; color: #64748b; }}
                .layout {{ display: grid; grid-template-columns: 2fr 1fr; gap: 22px; }}
                .model-card, .chart-card {{ background: white; border-radius: 16px; padding: 20px; margin: 0 0 22px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07); }}
                .model-card p, .chart-note {{ color: #64748b; line-height: 1.55; }}
                .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin-bottom: 26px; }}
                .bar-row, .metric-row {{ display: grid; grid-template-columns: minmax(110px, 1fr) minmax(100px, 2fr) 54px; align-items: center; gap: 12px; margin: 15px 0; font-size: 14px; }}
                .bar-track {{ height: 13px; overflow: hidden; background: #e2e8f0; border-radius: 999px; }}
                .bar {{ height: 100%; border-radius: inherit; }}
                .purpose-bar {{ background: #f59e0b; }}
                .metric-bar {{ background: #2563eb; }}
                .distribution {{ display: flex; height: 26px; overflow: hidden; border-radius: 999px; margin: 20px 0 12px; background: #e2e8f0; }}
                .distribution-safe {{ background: #10b981; }}
                .distribution-risk {{ background: #ef4444; }}
                .legend {{ display: flex; flex-wrap: wrap; gap: 16px; color: #475569; font-size: 14px; }}
                .legend span::before {{ content: ''; display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 7px; background: #10b981; }}
                .legend .risk-legend::before {{ background: #ef4444; }}
                .confusion {{ max-width: 520px; }}
                .confusion th, .confusion td {{ text-align: center; }}
                .note {{ margin: 18px 0; padding: 14px 16px; background: #eff6ff; border-radius: 10px; color: #334155; line-height: 1.5; }}
                table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06); }}
                th, td {{ padding: 12px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
                th {{ background: #eff6ff; }}
                @media (max-width: 700px) {{
                    .header {{ padding: 16px; align-items: flex-start; gap: 12px; flex-direction: column; }}
                    nav {{ flex-wrap: wrap; }}
                    .layout {{ grid-template-columns: 1fr; }}
                    .bar-row, .metric-row {{ grid-template-columns: minmax(90px, 1fr) minmax(70px, 2fr) 48px; gap: 8px; }}
                    .main {{ overflow-x: auto; }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2 style="margin: 0;">Loan Risk Portal</h2>
                <nav>
                    <a href="/dashboard">Dashboard</a>
                    <a href="/predict">Loan Prediction</a>
                    <a href="/logout">Logout ({username})</a>
                </nav>
            </div>

            <div class="main">
                <h1 style="margin-bottom: 24px;">Dashboard</h1>

                <section class="model-card" aria-labelledby="model-title">
                    <h2 id="model-title">{MODEL_NAME}</h2>
                    <p>A supervised classification model trained with {MODEL_TREES} trees. Performance below is measured on a held-out test set of {MODEL_EVALUATION['test_size']:,} applications (30% of the dataset); it is separate from the dashboard's historical application summaries.</p>
                    <strong>Estimated test accuracy: {MODEL_ACCURACY:.1%}</strong>
                </section>

                <div class="cards">
                    <div class="card">
                        <h3>Total Applications</h3>
                        <p class="value">{total_loans}</p>
                        <div class="sub">Across all loan records</div>
                    </div>
                    <div class="card">
                        <h3>Predicted Safe</h3>
                        <p class="value">{predicted_paid}</p>
                        <div class="sub">Likely fully paid</div>
                    </div>
                    <div class="card">
                        <h3>Predicted Risk</h3>
                        <p class="value">{predicted_default}</p>
                        <div class="sub">Likely not fully paid</div>
                    </div>
                    <div class="card">
                        <h3>Average Risk</h3>
                        <p class="value">{avg_risk:.2%}</p>
                        <div class="sub">Across the dataset</div>
                    </div>
                </div>

                <div class="charts">
                    <section class="chart-card" aria-labelledby="distribution-title">
                        <h2 id="distribution-title">Predicted risk distribution</h2>
                        <p class="chart-note">Model predictions across all {total_loans:,} historical applications.</p>
                        <div class="distribution" role="img" aria-label="{predicted_paid:,} predicted likely fully paid ({safe_share:.1f} percent), {predicted_default:,} predicted at risk ({risk_share:.1f} percent)">
                            <div class="distribution-safe" style="width:{safe_share:.2f}%"></div>
                            <div class="distribution-risk" style="width:{risk_share:.2f}%"></div>
                        </div>
                        <div class="legend">
                            <span>{predicted_paid:,} likely fully paid ({safe_share:.1f}%)</span>
                            <span class="risk-legend">{predicted_default:,} predicted at risk ({risk_share:.1f}%)</span>
                        </div>
                    </section>

                    <section class="chart-card" aria-labelledby="purpose-title">
                        <h2 id="purpose-title">Estimated risk by loan purpose</h2>
                        <p class="chart-note">Average predicted probability of not being fully paid.</p>
                        {purpose_bars}
                    </section>
                </div>

                <section class="chart-card" aria-labelledby="performance-title">
                    <h2 id="performance-title">Held-out model performance</h2>
                    <p class="chart-note">Metrics are calculated from predictions on test data that was not used to fit the model. Recall and precision refer to the at-risk class.</p>
                    <div class="charts">
                        <div>{metric_bars}</div>
                        <div>
                            <h3>Confusion matrix</h3>
                            <table class="confusion">
                                <thead><tr><th></th><th>Predicted paid</th><th>Predicted at risk</th></tr></thead>
                                <tbody>
                                    <tr><th>Actually paid</th><td>{true_paid:,}</td><td>{false_risk:,}</td></tr>
                                    <tr><th>Actually at risk</th><td>{missed_risk:,}</td><td>{detected_risk:,}</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </section>
                <p class="note"><strong>Important:</strong> These are model-based risk estimates for educational and informational purposes, not loan approvals or financial advice. The test metrics describe this dataset and do not guarantee future performance.</p>

                <div class="layout">
                    <div>
                        <h2>Recent predictions</h2>
                        <table>
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Purpose</th>
                                    <th>FICO</th>
                                    <th>Rate</th>
                                    <th>DTI</th>
                                    <th>Actual</th>
                                    <th>Prediction</th>
                                    <th>Risk</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join(rows)}
                            </tbody>
                        </table>
                    </div>

                    <div>
                        <h2>Risk by purpose</h2>
                        <table>
                            <thead>
                                <tr>
                                    <th>Purpose</th>
                                    <th>Avg risk</th>
                                </tr>
                            </thead>
                            <tbody>
                                {purpose_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </body>
        </html>
        '''
    )


def render_prediction_page(username: str) -> HTMLResponse:
    return HTMLResponse(
        f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>Predict | Loan Risk Prediction</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #f8fafc; margin: 0; color: #0f172a; }}
                .header {{ display: flex; justify-content: space-between; align-items: center; padding: 18px 32px; background: #0f172a; color: white; }}
                nav {{ display: flex; gap: 16px; align-items: center; }}
                nav a {{ color: white; text-decoration: none; font-weight: 600; }}
                .main {{ max-width: 900px; margin: 24px auto; padding: 0 18px 32px; }}
                form {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; background: white; border-radius: 16px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08); padding: 24px; }}
                .field {{ display: flex; flex-direction: column; gap: 8px; }}
                label {{ font-weight: 600; color: #334155; }}
                input, select {{ padding: 9px 10px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 14px; }}
                button {{ grid-column: 1 / -1; padding: 12px; border: none; border-radius: 10px; background: #2563eb; color: white; font-weight: 700; cursor: pointer; }}
                .box {{ margin-top: 20px; padding: 18px; background: #ecfdf5; border: 1px solid #86efac; border-radius: 12px; }}
                .error {{ background: #fef2f2; border-color: #fca5a5; color: #991b1b; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2 style="margin: 0;">Loan Risk Portal</h2>
                <nav>
                    <a href="/dashboard">Dashboard</a>
                    <a href="/predict">Loan Prediction</a>
                    <a href="/logout">Logout ({username})</a>
                </nav>
            </div>

            <div class="main">
                <h1>Loan Risk Prediction</h1>
                <form id="loan-form">
                    <div class="field">
                        <label>Credit policy</label>
                        <select name="credit_policy">
                            <option value="0">0</option>
                            <option value="1" selected>1</option>
                        </select>
                    </div>
                    <div class="field">
                        <label>Purpose</label>
                        <select name="purpose">
                            {''.join(f'<option value="{p}">{p}</option>' for p in PURPOSES)}
                        </select>
                    </div>
                    <div class="field"><label>Interest rate (%)</label><input name="int_rate" type="number" value="12" min="5" max="25" step="0.1" /></div>
                    <div class="field"><label>Installment</label><input name="installment" type="number" value="300" min="50" max="1200" step="0.01" /></div>
                    <div class="field"><label>Log annual income</label><input name="log_annual_inc" type="number" value="10.5" min="7" max="14" step="0.1" /></div>
                    <div class="field"><label>DTI</label><input name="dti" type="number" value="12" min="0" max="40" step="0.1" /></div>
                    <div class="field"><label>FICO score</label><input name="fico" type="number" value="700" min="300" max="850" step="1" /></div>
                    <div class="field"><label>Days with credit line</label><input name="days_with_cr_line" type="number" value="1000" min="0" max="10000" step="1" /></div>
                    <div class="field"><label>Revolving balance</label><input name="revol_bal" type="number" value="15000" min="0" max="200000" step="1" /></div>
                    <div class="field"><label>Revolving utilization (%)</label><input name="revol_util" type="number" value="50" min="0" max="100" step="0.1" /></div>
                    <div class="field"><label>Inquiries last 6 months</label><input name="inq_last_6mths" type="number" value="1" min="0" max="10" step="1" /></div>
                    <div class="field"><label>Delinquencies past 2 years</label><input name="delinq_2yrs" type="number" value="0" min="0" max="10" step="1" /></div>
                    <div class="field"><label>Public records</label><input name="pub_rec" type="number" value="0" min="0" max="10" step="1" /></div>
                    <button type="submit">Predict Loan Outcome</button>
                </form>
                <div id="result" class="box" style="display:none;"></div>
                <div id="error" class="box error" style="display:none;"></div>
            </div>

            <script>
                const form = document.getElementById('loan-form');
                const resultBox = document.getElementById('result');
                const errorBox = document.getElementById('error');
                form.addEventListener('submit', async (event) => {{
                    event.preventDefault();
                    const formData = new FormData(form);
                    const payload = Object.fromEntries(formData.entries());
                    payload.credit_policy = Number(payload.credit_policy);
                    payload.int_rate = Number(payload.int_rate);
                    payload.installment = Number(payload.installment);
                    payload.log_annual_inc = Number(payload.log_annual_inc);
                    payload.dti = Number(payload.dti);
                    payload.fico = Number(payload.fico);
                    payload.days_with_cr_line = Number(payload.days_with_cr_line);
                    payload.revol_bal = Number(payload.revol_bal);
                    payload.revol_util = Number(payload.revol_util);
                    payload.inq_last_6mths = Number(payload.inq_last_6mths);
                    payload.delinq_2yrs = Number(payload.delinq_2yrs);
                    payload.pub_rec = Number(payload.pub_rec);

                    try {{
                        const response = await fetch('/api/predict', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify(payload),
                        }});
                        const data = await response.json();
                        if (!response.ok) {{
                            throw new Error(data.detail || 'Prediction failed');
                        }}
                        errorBox.style.display = 'none';
                        resultBox.innerHTML = '<strong>' + data.label + '</strong><div>Probability of not fully paid: ' + (data.probability_not_fully_paid * 100).toFixed(2) + '%</div><div>Probability of fully paid: ' + (data.probability_fully_paid * 100).toFixed(2) + '%</div>';
                        resultBox.style.display = 'block';
                    }} catch (error) {{
                        errorBox.textContent = error.message;
                        errorBox.style.display = 'block';
                        resultBox.style.display = 'none';
                    }}
                }});
            </script>
        </body>
        </html>
        '''
    )


@app.get('/')
def home(request: Request):
    user = get_logged_in_user(request)
    if user:
        return RedirectResponse('/dashboard', status_code=302)
    return RedirectResponse('/login', status_code=302)


@app.get('/login')
def login_page(request: Request):
    if get_logged_in_user(request):
        return RedirectResponse('/dashboard', status_code=302)
    return render_login_page()


@app.post('/login')
def do_login(username: str = Form(...), password: str = Form(...)):
    if authenticate_user(username, password):
        response = RedirectResponse('/dashboard', status_code=302)
        set_session_cookie(response, username.strip().lower())
        return response
    return render_login_page('Invalid username or password.')


@app.get('/register')
def register_page(request: Request):
    if get_logged_in_user(request):
        return RedirectResponse('/dashboard', status_code=302)
    return render_register_page()


@app.post('/register')
def register(
    full_name: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    full_name = full_name.strip()
    username = username.strip().lower()

    if not full_name or len(full_name) > 100:
        return render_register_page('Enter a name of 1 to 100 characters.')
    if not re.fullmatch(r'[a-zA-Z0-9_.-]{3,32}', username):
        return render_register_page(
            'Username must be 3–32 characters and use only letters, numbers, dots, underscores, or hyphens.'
        )
    if username in AUTH_USERS:
        return render_register_page('That username is already reserved.')
    if len(password) < 8:
        return render_register_page('Password must be at least 8 characters.')
    if password != confirm_password:
        return render_register_page('Passwords do not match.')

    salt, password_hash = hash_password(password)
    try:
        with sqlite3.connect(USER_DATABASE) as connection:
            connection.execute(
                'INSERT INTO users (username, full_name, password_salt, password_hash) VALUES (?, ?, ?, ?)',
                (username, full_name, salt, password_hash),
            )
    except sqlite3.IntegrityError:
        return render_register_page('That username is already registered.')

    response = RedirectResponse('/dashboard', status_code=302)
    set_session_cookie(response, username)
    return response


@app.get('/logout')
def logout():
    response = RedirectResponse('/login', status_code=302)
    clear_session_cookie(response)
    return response


@app.get('/dashboard')
def dashboard(request: Request):
    user = get_logged_in_user(request)
    if not user:
        return RedirectResponse('/login', status_code=302)
    return render_dashboard_page(user)


@app.get('/predict')
def predict_page(request: Request):
    user = get_logged_in_user(request)
    if not user:
        return RedirectResponse('/login', status_code=302)
    return render_prediction_page(user)


@app.post('/api/predict')
async def api_predict(request: Request):
    user = get_logged_in_user(request)
    if not user:
        raise HTTPException(status_code=401, detail='Authentication required')

    try:
        payload = await request.json()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail='Invalid JSON payload') from exc

    try:
        result = predict_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result
