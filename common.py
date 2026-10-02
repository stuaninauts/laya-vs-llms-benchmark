"""Definições compartilhadas: filas de produto, pergunta de fraude, mapeamentos da base CFPB."""
import re

MAX_CHARS = 1500  # mesmo texto truncado para Laya e para o LLM

# fila -> descrição (vira `criteria` no Laya e lista de opções no prompt do LLM)
PRODUCTS = {
    "credit_reporting": "credit reports, credit scores, credit bureaus, disputes of report information",
    "debt_collection": "debt collectors, collection calls or letters, debts the consumer says are not owed",
    "credit_card": "credit cards: charges, billing disputes, interest, card account management",
    "checking_savings": "checking or savings bank accounts: deposits, withdrawals, overdrafts, account access",
    "money_transfer": "money transfers, wires, payment apps, virtual currency, money services",
    "mortgage": "mortgages and home loans: payments, escrow, servicing, foreclosure, refinancing",
    "vehicle_loan": "auto loans and vehicle leases",
    "student_loan": "federal or private student loans and their servicers",
    "personal_loan": "payday loans, title loans, personal loans, cash advances",
}

PRODUCT_QUESTION = {
    "type": "choice",
    "instructions": "Which team should handle this consumer complaint?",
    "criteria": PRODUCTS,
}

FRAUD_QUESTION = {
    "type": "noul",
    "instructions": "Does the consumer report fraud, a scam or identity theft against them?",
    "criteria": {"false": "no fraud, scam or identity theft reported",
                 "true": "reports fraud, a scam or identity theft"},
}

QUESTIONS = {"product": PRODUCT_QUESTION, "fraud": FRAUD_QUESTION}


def map_product(product: str, sub_product: str | None) -> str | None:
    p, s = product.lower(), (sub_product or "").lower()
    if "prepaid" in s or p == "prepaid card" or p == "debt or credit management":
        return None
    if p.startswith("credit reporting"):
        return "credit_reporting"
    if p == "debt collection":
        return "debt_collection"
    if p.startswith("credit card"):
        return "credit_card"
    if p == "checking or savings account":
        return "checking_savings"
    if p.startswith("money transfer"):
        return "money_transfer"
    if p == "mortgage":
        return "mortgage"
    if p == "vehicle loan or lease":
        return "vehicle_loan"
    if p == "student loan":
        return "student_loan"
    if p.startswith("payday loan"):
        return "personal_loan"
    return None


FRAUD_POS = re.compile(r"fraud or scam|result of identity theft|as result of identity theft or fraud|"
                       r"opened as a result of fraud|fraudulent loan|unauthorized withdrawals or charges", re.I)
# rótulos ambíguos: nem claramente fraude nem claramente não-fraude -> fora do experimento
FRAUD_AMBIGUOUS = re.compile(r"belongs to someone else|fraud alerts|identity theft protection|"
                             r"identify theft protection|monitoring services|unauthorized transactions or other|"
                             r"improper use of your report", re.I)


def fraud_label(issue: str | None, sub_issue: str | None) -> bool | None:
    txt = f"{issue or ''} | {sub_issue or ''}"
    if FRAUD_POS.search(txt):
        return True
    if FRAUD_AMBIGUOUS.search(txt):
        return None
    return False
