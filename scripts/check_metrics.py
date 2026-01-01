import json
import sys

THRESHOLDS = {
    "recall": 0.75,
    "mrr": 0.55,
    "exact_match": 0.55,
    "token_f1": 0.70,
}

data = json.loads(sys.stdin.read())

print("Evaluation Metrics:", data)

failed = False

for metric, min_val in THRESHOLDS.items():
    score = data.get(metric)

    if score is None:
        print(f"❌ Missing metric: {metric}")
        failed = True
        continue

    if score < min_val:
        print(f"❌ {metric}={score:.3f} < required {min_val}")
        failed = True
    else:
        print(f"✅ {metric}={score:.3f}")

if failed:
    sys.exit(1)
else:
    print("\n🎉 Metrics look good!")