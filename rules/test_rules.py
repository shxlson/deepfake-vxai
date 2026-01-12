from rules.rules import generate_explanation

# Simulated values (fake-like case)
text = generate_explanation(
    activation_coverage=0.32,
    fake_similarity=0.83,
    real_similarity=0.21,
    activated_frames=7
)

print("Explanation:")
print(text)
