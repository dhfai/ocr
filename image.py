from PIL import Image

# Buka gambar
image = Image.open("data/dhf.jpg")

# Ubah gambar menjadi hitam putih
bw_image = image.convert("L")

# Simpan gambar hasil
bw_image.save("gen.jpg")

# Atau, jika ingin menampilkan hasilnya
bw_image.show()
