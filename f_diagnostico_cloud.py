import pandas as pd
import numpy as np

# ============================================================
# DIAGNOSTICO DE cloud_indicator
# Copia y pega EXACTAMENTE esto en tu Jupyter/Spyder
# ============================================================

# PASO 1: Carga los datos
archivo_excel = r'C:\Users\Usuario\Desktop\Trabajo_UNI\Gemelos digitales\Articulos\ART_2\fINAL_REV\Datos temperatura_2_79 kW.xlsx'
df = pd.read_excel(archivo_excel, sheet_name='Hoja1')

# PASO 2: Mapeo de columnas
column_mapping = {
    'Fecha/Hora': 'datetime',
    'Temperatura (ºC)': 'T_amb',
    'Irradiancia (W/m2)': 'G', 
    'Wind speed (m/s)': 'w',
    'Potencia real (kW)': 'P_real',
    'TONC': 'NOCT',
    'Tref (ºC)': 'T_ref',
    'Pref,mpp (W)': 'P_ref'
}

for old_name, new_name in column_mapping.items():
    if old_name in df.columns:
        df[new_name] = df[old_name]

# Elimina valores nulos
df = df.dropna(subset=['T_amb', 'G', 'w', 'P_real'])

if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])

# PASO 3: Calcula variables avanzadas
df['dG'] = df['G'].diff().fillna(0)
df['sigma_G_5'] = df['G'].rolling(window=5, min_periods=1).std().fillna(0)

# PASO 4: DIAGNÓSTICO CON UMBRALES ACTUALES
print("\n" + "="*80)
print("DIAGNÓSTICO DE cloud_indicator")
print("="*80)

cloud_current = ((df['dG'].abs() > 50) & (df['sigma_G_5'] > 100)).astype(float)
pct_current = cloud_current.sum() / len(df) * 100

print(f"\n┌─ UMBRALES ACTUALES (dG > 50, σ_G > 100) ──────────────┐")
print(f"│ Total observaciones: {len(df):,}")
print(f"│ Eventos cloud_indicator = 1: {cloud_current.sum():.0f}")
print(f"│ Porcentaje: {pct_current:.2f}%")
print("└──────────────────────────────────────────────────────────┘")

# Interpretación
if pct_current < 1:
    interpretacion = "❌ CRÍTICO - Casi nunca se activa"
elif pct_current < 5:
    interpretacion = "⚠️ BAJO - El revisor tiene razón, cambiar umbrales"
elif pct_current < 10:
    interpretacion = "⚠️ Marginal - Considerar cambiar umbrales"
elif pct_current < 30:
    interpretacion = "✓ BIEN - Umbrales razonables"
else:
    interpretacion = "✓ ALTO - Umbrales sensibles, posiblemente falsos positivos"

print(f"\n│ Interpretación: {interpretacion}")

# PASO 5: PROPUESTA DE NUEVOS UMBRALES
print(f"\n┌─ PROPUESTA: UMBRALES MÁS BAJOS (dG > 30, σ_G > 50) ────┐")

cloud_proposed = ((df['dG'].abs() > 30) & (df['sigma_G_5'] > 50)).astype(float)
pct_proposed = cloud_proposed.sum() / len(df) * 100

print(f"│ Eventos cloud_indicator = 1: {cloud_proposed.sum():.0f}")
print(f"│ Porcentaje: {pct_proposed:.2f}%")

if 10 < pct_proposed < 30:
    interpretacion_new = "✓ EXCELENTE - Estos umbrales funcionan"
elif pct_proposed < 10:
    interpretacion_new = "⚠️ Aún bajo - Considerar umbrales más bajos"
elif pct_proposed > 30:
    interpretacion_new = "⚠️ Alto - Falsos positivos, umbrales muy bajos"
else:
    interpretacion_new = "? A evaluar"

print(f"│ Interpretación: {interpretacion_new}")
print("└──────────────────────────────────────────────────────────┘")

# PASO 6: ESTADÍSTICAS DE LAS VARIABLES
print(f"\n┌─ ESTADÍSTICAS DE dG (cambio irradiancia) ──────────────┐")
print(f"│ Máximo: {df['dG'].abs().max():.2f} W/m²")
print(f"│ Percentil 95: {df['dG'].abs().quantile(0.95):.2f} W/m²")
print(f"│ Percentil 90: {df['dG'].abs().quantile(0.90):.2f} W/m²")
print(f"│ Percentil 75: {df['dG'].abs().quantile(0.75):.2f} W/m²")
print(f"│ Percentil 50 (mediana): {df['dG'].abs().quantile(0.50):.2f} W/m²")
print(f"│ Tu umbral actual: 50 W/m²")
print("└──────────────────────────────────────────────────────────┘")

print(f"\n┌─ ESTADÍSTICAS DE σ_G (variabilidad irradiancia) ────────┐")
print(f"│ Máximo: {df['sigma_G_5'].max():.2f} W/m²")
print(f"│ Percentil 95: {df['sigma_G_5'].quantile(0.95):.2f} W/m²")
print(f"│ Percentil 90: {df['sigma_G_5'].quantile(0.90):.2f} W/m²")
print(f"│ Percentil 75: {df['sigma_G_5'].quantile(0.75):.2f} W/m²")
print(f"│ Percentil 50 (mediana): {df['sigma_G_5'].quantile(0.50):.2f} W/m²")
print(f"│ Tu umbral actual: 100 W/m²")
print("└──────────────────────────────────────────────────────────┘")

# PASO 7: COMPARACIÓN DE OPCIONES
print(f"\n┌─ COMPARACIÓN DE OPCIONES DE UMBRALES ──────────────────┐")

opciones = [
    ("Actual", 50, 100),
    ("Opción 1", 30, 50),
    ("Opción 2", 20, 30),
    ("Opción 3 (Muy sensible)", 15, 20),
]

print(f"│ {'Nombre':<20} {'dG':<8} {'σ_G':<8} {'Eventos':<10} {'%':<8}")
print("├───────────────────────────────────────────────────────────┤")

for nombre, dg_thresh, sigma_thresh in opciones:
    cloud_option = ((df['dG'].abs() > dg_thresh) & (df['sigma_G_5'] > sigma_thresh)).astype(float)
    pct_option = cloud_option.sum() / len(df) * 100
    print(f"│ {nombre:<20} {dg_thresh:<8} {sigma_thresh:<8} {cloud_option.sum():<10.0f} {pct_option:<8.2f}%")

print("└───────────────────────────────────────────────────────────┘")

# PASO 8: RECOMENDACIÓN FINAL
print(f"\n┌─ RECOMENDACIÓN FINAL ─────────────────────────────────┐")

if pct_current < 5:
    print(f"│ ✓ EL REVISOR TIENE RAZÓN")
    print(f"│")
    print(f"│ Acción recomendada:")
    print(f"│ 1. Cambia umbrales en calcular_variables_avanzadas():")
    print(f"│")
    print(f"│    ANTES:")
    print(f"│    cloud_indicator = ((dG.abs() > 50) & (sigma_G_5 > 100))")
    print(f"│")
    print(f"│    DESPUÉS:")
    print(f"│    cloud_indicator = ((dG.abs() > 30) & (sigma_G_5 > 50))")
    print(f"│")
    print(f"│ 2. Recalibra el modelo")
    print(f"│ 3. Compara nuevo R² con el anterior")
    print(f"│ 4. Responde al revisor con los nuevos números")
elif pct_current > 10:
    print(f"│ ✗ EL REVISOR PROBABLEMENTE SE EQUIVOCÓ")
    print(f"│")
    print(f"│ cloud_indicator está activo {pct_current:.2f}% del tiempo")
    print(f"│ Los umbrales son razonables.")
    print(f"│")
    print(f"│ Responde al revisor:")
    print(f"│ 'He verificado experimentalmente que cloud_indicator")
    print(f"│  se activa {pct_current:.2f}% del tiempo, lo que es razonable'")
else:
    print(f"│ ? CASO MARGINAL")
    print(f"│")
    print(f"│ Porcentaje: {pct_current:.2f}%")
    print(f"│ Considera cambiar umbrales de todas formas para mayor robustez")

print("└───────────────────────────────────────────────────────────┘")

# RESUMEN FINAL
print(f"\n" + "="*80)
print("RESUMEN PARA COMPARTIR")
print("="*80)
print(f"\ncloud_indicator activado: {pct_current:.2f}% del tiempo")
print(f"Propuesta con nuevos umbrales: {pct_proposed:.2f}%")
print(f"\n¿Necesita cambio? {'SÍ' if pct_current < 5 else 'NO'}")
print("="*80)