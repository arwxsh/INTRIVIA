from Burnout_model import BurnoutModel
import pandas as pd


model = BurnoutModel()
model.train("Data Carrard et al. 2022 MedTeach.csv")


model.load()


new_data = pd.DataFrame([{
    'sex': int(input("Pilihan jenis kelamin: \n1=Man; 2=Woman; 3=Non-binary;\nMasukan jenis kelamin: ")), 
    'age': float(input("Masukan umur dalam tahun: ")), 
    'year': float(input("Masukan tahun kuliah ke-: ")), 
    'stud_h': float(input("Berapa jam anda belajar dalam 1 minggu: ")), 
    'part': int(input("Apakah anda memiliki pasangan?\n1=Ya; 0=Tidak;\nMasukan pilihan: ")), 
    'job': int(input("Apakah anda sedang bekerja?\n1=Ya; 0=Tidak;\nMasukan pilihan: ")), 
    'health': int(input("Seberapa puaskah Anda dengan kesehatan Anda?\n1=Sangat tidak puas; 2=Tidak puas; 3=Biasa saja; 4=Puas; 5=Sangat puas\nMasukan pilihan: "))
}])


burnout_preds = model.predict_burnout(new_data)


print("\n" + "="*60)
print("HASIL PREDIKSI BURNOUT")
print("="*60)

print("\n1. MBI-EX (Emotional Exhaustion) Score:", f"{burnout_preds['mbi_ex'][0]:.2f}")
print("   Interpretasi:")
print("   0-16    : Low exhaustion")
print("   17-26   : Moderate exhaustion")
print("   27+     : High exhaustion (Burnout)")

print("\n2. MBI-CY (Cynicism) Score:", f"{burnout_preds['mbi_cy'][0]:.2f}")
print("   Interpretasi:")
print("   0-6     : Low cynicism")
print("   7-12    : Moderate cynicism")
print("   13+     : High cynicism (Disengagement)")

print("\n3. MBI-EA (Academic Efficacy) Score:", f"{burnout_preds['mbi_ea'][0]:.2f}")
print("   Interpretasi:")
print("   0-30    : Low efficacy (Reduced accomplishment)")
print("   31-36   : Moderate efficacy")
print("   37+     : High efficacy (Effective)")

print("\n" + "="*60)
print("OVERALL BURNOUT ASSESSMENT")
print("="*60)

ex_score = burnout_preds['mbi_ex'][0]
cy_score = burnout_preds['mbi_cy'][0]
ea_score = burnout_preds['mbi_ea'][0]

has_high_ex = ex_score >= 17
has_high_cy = cy_score >= 7
has_low_ea = ea_score <= 30

if has_high_ex and (has_high_cy or has_low_ea):
    print("Status: BURNOUT DETECTED")
    print("Anda menunjukkan tanda-tanda burnout yang signifikan.")
    print("Disarankan untuk mencari dukungan profesional.")
elif has_high_ex:
    print("Status: OVEREXTENDED")
    print("Anda mengalami kelelahan tinggi tetapi masih terlibat dengan pekerjaan/studi.")
elif has_high_cy:
    print("Status: DISENGAGED")
    print("Anda mengalami sinisme/jarak emosional terhadap pekerjaan/studi.")
elif has_low_ea:
    print("Status: INEFFECTIVE")
    print("Anda merasa kurang efektif dalam mencapai tujuan.")
else:
    print("Status: ENGAGED")
    print("Anda menunjukkan tingkat keterlibatan yang baik dengan tingkat burnout rendah.")

print("="*60)
