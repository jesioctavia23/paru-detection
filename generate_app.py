import os

folder = "./flask_app_final"
os.makedirs(folder, exist_ok=True)

main_code_path = os.path.join(folder, "app.py")
template_path = os.path.join(folder, "templates")
static_path = os.path.join(folder, "static")
upload_path = os.path.join(folder, "uploads")

# Buat folder struktur
os.makedirs(template_path, exist_ok=True)
os.makedirs(static_path, exist_ok=True)
os.makedirs(upload_path, exist_ok=True)

# Isi script Flask lengkap
final_code = """\
from flask import Flask, request, render_template
# (lanjutkan isi app seperti yang sudah kamu punya...)
"""

with open(main_code_path, "w") as f:
    f.write(final_code)
