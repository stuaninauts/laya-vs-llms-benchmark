"""Amostras fixas do conjunto de teste para os LLMs mais caros (sem custo, determinístico).

- test_s900:   100 por fila (metade do teste), mesma taxa de fraude por fila do teste completo.
               Todos os sistemas também são avaliados nesta amostra → comparação pareada justa.
- test_lat200: 200 reclamações (≈22 por fila) contidas na s900, só para medir latência síncrona;
               a acurácia dos modelos caros vem do Batch API (50% mais barato, sem latência útil).
"""
import pandas as pd

SEED = 20260930
test = pd.read_parquet("data/test.parquet")

s900 = (test.groupby(["queue", "fraud"], group_keys=False)
        .apply(lambda g: g.sample(frac=0.5, random_state=SEED)))
# ajuste fino para exatamente 100 por fila (arredondamento dos estratos de fraude)
s900 = s900.groupby("queue", group_keys=False).apply(lambda g: g.head(100) if len(g) >= 100 else g)
missing = {q: 100 - n for q, n in s900.queue.value_counts().items() if n < 100}
for q, k in missing.items():
    extra = test[(test.queue == q) & ~test.id.isin(s900.id)].sample(k, random_state=SEED)
    s900 = pd.concat([s900, extra])
s900 = s900.sample(frac=1, random_state=SEED)

lat200 = s900.groupby("queue", group_keys=False).apply(lambda g: g.sample(22, random_state=SEED)).head(200)

s900.to_parquet("data/test_s900.parquet")
lat200.to_parquet("data/test_lat200.parquet")
print(f"s900: {len(s900)} | fraude {s900.fraud.mean():.1%} (teste completo {test.fraud.mean():.1%}) | "
      f"por fila {s900.queue.value_counts().min()}–{s900.queue.value_counts().max()}")
print(f"lat200: {len(lat200)} | fraude {lat200.fraud.mean():.1%} | contida na s900: {lat200.id.isin(s900.id).all()}")
