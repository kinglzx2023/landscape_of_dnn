import subprocess

pros = [0.0, 0.2, 0.3, 0.4, 0.5]

# 循环调用 train_network.py 并传入不同的随机种子
for i, pro in enumerate(pros):
    print(f"Running train_network.py with seed {pro} (Attempt {i + 1})")
    subprocess.run(["python", "dropout_adamw_sst2_bert.py", str(pro)])