"""Prompt, schema e tabela de preços compartilhados por todos os LLMs de API.

O SYSTEM e o SCHEMA são os mesmos usados no baseline já pago (gpt-5.4-nano/mini), para que todo
LLM de API receba exatamente a mesma instrução — diferença de prompt não entra como variável.
"""
from common import PRODUCTS, FRAUD_QUESTION

SYSTEM = (
    "You triage consumer complaints received by a US financial institution.\n"
    "Return JSON only.\n"
    "product: the queue that must handle the complaint. Options:\n"
    + "\n".join(f"- {k}: {v}" for k, v in PRODUCTS.items())
    + f"\nfraud: {FRAUD_QUESTION['instructions']} "
    f"(true = {FRAUD_QUESTION['criteria']['true']}; false = {FRAUD_QUESTION['criteria']['false']})\n"
    "fraud_probability: your probability (0-1) that fraud is true."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "product": {"type": "string", "enum": list(PRODUCTS)},
        "fraud": {"type": "boolean"},
        "fraud_probability": {"type": "number"},
    },
    "required": ["product", "fraud", "fraud_probability"],
    "additionalProperties": False,
}

# US$ por 1M tokens (entrada, saída), preço de lista — consultado em 30/09/2026.
# OpenAI: tabelas públicas (morphllm / benchlm, set/2026). Anthropic: referência oficial do SDK.
# Batch API: 50% de desconto nos dois provedores.
PRICES = {
    ("openai", "gpt-5.4-nano"): (0.20, 1.25),
    ("openai", "gpt-5.4-mini"): (0.75, 4.50),
    ("openai", "gpt-5.6-luna"): (0.20, 1.20),
    ("openai", "gpt-5.6-terra"): (2.00, 12.00),
    ("openai", "gpt-5.6-sol"): (5.00, 30.00),     # promo US$ 4/20 até nov/2026; usamos o de lista
    ("openai", "gpt-5.5"): (5.00, 30.00),
    ("openai", "gpt-4.1-nano"): (0.10, 0.40),
    ("openai", "gpt-4.1-mini"): (0.40, 1.60),
    ("anthropic", "claude-haiku-4-5"): (1.00, 5.00),
    ("anthropic", "claude-sonnet-5-5"): (2.00, 10.00),
    ("anthropic", "claude-opus-5-5"): (4.00, 20.00),
}
BATCH_DISCOUNT = 0.5

# Tokens por reclamação medidos no baseline gpt-5.4-nano (1.800 chamadas): 517,6 entrada / 26,4 saída.
# Anthropic usa outro tokenizer: estimamos +20% (confirmar com count_tokens, que é gratuito).
# Opus 5.5 não desliga o raciocínio: saída estimada bem maior.
TOKENS_IN = {"openai": 518, "anthropic": 622}
TOKENS_OUT = {"default": 30, "claude-opus-5-5": 250}


def estimate(provider, model, n, batch):
    pin, pout = PRICES[(provider, model)]
    tout = TOKENS_OUT.get(model, TOKENS_OUT["default"])
    usd = n * (TOKENS_IN[provider] * pin + tout * pout) / 1e6
    return usd * (BATCH_DISCOUNT if batch else 1.0)
