import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



# === CONFIGURACIÓN GLOBAL DE MATPLOTLIB PARA CALIDAD Q1 ===
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.minor.width'] = 0.8
plt.rcParams['ytick.minor.width'] = 0.8
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['lines.linewidth'] = 1.8
plt.rcParams['lines.markersize'] = 4
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False



# === 1. PARÁMETROS DE ENTRADA (¡SOLO CAMBIA ESTOS DOS!) ===
estacion_seleccionada = 'Otoño'  # Opciones: 'Verano', 'Otoño', 'Invierno', 'Primavera'
dia_seleccionado = 23        # Día del mes a analizar
mes_seleccionado = 11            # Mes a analizar (ejemplo: 6 para junio, 9 para septiembre, etc.)



# === 2. CARGAR PARÁMETROS CALIBRADOS ===
with open('parametros_calibrados.json', 'r') as f:
    parametros_por_estacion = json.load(f)
params = parametros_por_estacion[estacion_seleccionada]



# === 3. LEER DATOS DEL EXCEL ===
archivo_excel = r'C:\Users\Usuario\Desktop\Trabajo_UNI\Gemelos digitales\Articulos\ART_2\fINAL_REV\Datos temperatura_2_79 kW.xlsx'
df = pd.read_excel(archivo_excel, sheet_name='Hoja2')
df['datetime'] = pd.to_datetime(df['Fecha/Hora'])
df_sel = df[(df['datetime'].dt.month == mes_seleccionado) & (df['datetime'].dt.day == dia_seleccionado)].copy()



# Renombrar columnas para compatibilidad
for col_old, col_new in [('Temperatura (ºC)', 'T_amb'), ('Irradiancia (W/m2)', 'G'), ('Wind speed (m/s)', 'w')]:
    if col_old in df_sel.columns:
        df_sel[col_new] = df_sel[col_old]



# === 4. CALCULAR VARIABLES AVANZADAS ===
def calcular_variables_avanzadas(df):
    df = df.copy()
    df['dG'] = df['G'].diff().fillna(0)
    df['d2G'] = df['dG'].diff().fillna(0)
    df['sigma_G_10'] = df['G'].rolling(window=10, min_periods=1).std().fillna(0)
    df['cloud_indicator'] = ((df['dG'].abs() > 50) & (df['sigma_G_10'] > 100)).astype(float)
    df['w_squared'] = df['w'] ** 2
    df['G_sqrt'] = np.sqrt(df['G'].clip(lower=0))
    df['sigma_G_5'] = df['sigma_G_10']
    return df



df_prep = calcular_variables_avanzadas(df_sel)



# === 5. MODELO ULTRA MEJORADO ===
def modelo_ultra_mejorado(df, params, NOCT=45.0, T_ref=25.0):
    (alpha1, alpha2, alpha3, beta1, beta2, beta3, gamma1, gamma2, gamma3, delta, C_med) = params
    n = len(df)
    T_cell = np.zeros(n)
    T_amb = df['T_amb'].values
    G = df['G'].values
    w = df['w'].values
    dG = df['dG'].values
    d2G = df['d2G'].values
    sigma_G_5 = df['sigma_G_5'].values
    cloud_ind = df['cloud_indicator'].values
    w_sq = df['w_squared'].values
    G_sqrt = df['G_sqrt'].values



    T_MIN, T_MAX = -10, 80
    INERCIA_MIN, INERCIA_MAX = -50, 50



    for i in range(n):
        T_amb_i = np.clip(T_amb[i], T_MIN, T_MAX)
        base = T_amb_i + (NOCT - 20)/800 * G[i]
        inercia = 0
        if i >= 1:
            diff1 = np.clip(T_cell[i-1], T_MIN, T_MAX) - T_amb_i
            inercia += alpha1 * diff1
        if i >= 2:
            diff2 = np.clip(T_cell[i-2], T_MIN, T_MAX) - T_amb_i
            inercia += alpha2 * diff2
        if i >= 3:
            diff3 = np.clip(T_cell[i-3], T_MIN, T_MAX) - T_amb_i
            inercia += alpha3 * diff3
        inercia = np.clip(inercia, INERCIA_MIN, INERCIA_MAX)
        viento = (beta1 * (T_amb_i - T_ref) * w[i] +
                  beta2 * dG[i] * w[i] +
                  beta3 * w_sq[i] * (T_amb_i - T_ref))
        nubosidad = (gamma1 * abs(dG[i]) +
                     gamma2 * sigma_G_5[i] * cloud_ind[i] +
                     gamma3 * d2G[i] * cloud_ind[i])
        no_lineal = delta * G_sqrt[i] * (T_amb_i - T_ref)
        T_cell[i] = base + inercia + viento + nubosidad + no_lineal + C_med
        T_cell[i] = np.clip(T_cell[i], T_MIN, T_MAX)
    return T_cell



T_cell = modelo_ultra_mejorado(df_prep, params)



# === 6. MODELOS DE TEMPERATURA DE CÉLULA ===
NOCT = 45.0
T_ref = 25.0
P_ref = 465.0
num_paneles = 6
beta_temp = 0.004



# NOCT clásico
T_cell_NOCT = df_prep['T_amb'] + (NOCT - 20)/800 * df_prep['G']

# King 2004
T_cell_King = df_prep['T_amb'] + (df_prep['G']/1000)*(NOCT-20) / (1 + 0.025*df_prep['w'])

# Skoplaki & Palyvos 2009
eta = 0.15
T_cell_Skoplaki = df_prep['T_amb'] + (df_prep['G']/800)*(NOCT-20) - eta*(df_prep['G']/800)

# PVsyst/Faiman
U0 = 25.0
U1 = 6.84
T_cell_Faiman = df_prep['T_amb'] + df_prep['G'] / (U0 + U1*df_prep['w'])



# === 7. FUNCIÓN GENERAL DE POTENCIA PV ===
def calcular_potencia_hong(T_cell, df, P_ref=465.0, T_ref=25.0):
    beta_temp = 0.004
    num_paneles = 6
    power = (P_ref * (df['G'] / 1000) * (1 - beta_temp * (T_cell - T_ref)) * num_paneles)
    return power / 1000  # kW



# === 8. CALCULAR POTENCIA PV PARA CADA MODELO ===
P_simulada_ultra = calcular_potencia_hong(T_cell, df_prep)
P_simulada_NOCT = calcular_potencia_hong(T_cell_NOCT, df_prep)
P_simulada_King = calcular_potencia_hong(T_cell_King, df_prep)
P_simulada_Skoplaki = calcular_potencia_hong(T_cell_Skoplaki, df_prep)
P_simulada_Faiman = calcular_potencia_hong(T_cell_Faiman, df_prep)



# === 9. GRAFICAR CON CALIDAD Q1 (MEJORADO) ===
fig, ax = plt.subplots(figsize=(14, 7.5), dpi=100)

# NO usar constrained_layout, usar subplots_adjust
fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.13)

# Potencia real 
ax.plot(df_prep['datetime'], df_prep['Potencia real (kW)'], 
        label='Measured Power', color='black', linestyle='-', 
        linewidth=2.2, zorder=5, marker='o', markersize=3.5, markevery=6)

# Modelo personal 
ax.plot(df_prep['datetime'], P_simulada_ultra, 
        label='Proposed Model', color='#0066cc', linestyle='-', 
        linewidth=2, zorder=4, alpha=0.95)

# Modelos de referencia
ax.plot(df_prep['datetime'], P_simulada_NOCT, 
        label='NOCT', color='#00aa00', linestyle='--', linewidth=2.8, alpha=0.95, zorder=4)
ax.plot(df_prep['datetime'], P_simulada_King, 
        label='King', color='#ff8800', linestyle='--', linewidth=2, alpha=0.95, zorder=4)
ax.plot(df_prep['datetime'], P_simulada_Skoplaki, 
        label='Skoplaki', color='#aa33ff', linestyle='--', linewidth=2, alpha=0.95, zorder=4)
ax.plot(df_prep['datetime'], P_simulada_Faiman, 
        label='PVsyst/Faiman', color='#ff3333', linestyle='--', linewidth=1.8, alpha=0.95, zorder=4)

# Formateo de ejes X
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))

# Rotación optimizada
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=15)

# Etiquetas con mejor espaciado
ax.set_xlabel('Time (h)', fontsize=17, fontweight='bold', labelpad=8)
ax.set_ylabel('Power output (kW)', fontsize=17, fontweight='bold', labelpad=8)

# TÍTULO CORREGIDO (dividido en 2 líneas)
ax.set_title(f'Comparison of simulated and real PV power ({dia_seleccionado}/{mes_seleccionado})', 
             fontsize=19, fontweight='bold', pad=15, loc='center')

# Grid limpio para Q1
ax.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.35, color='gray', zorder=0)
ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.15, color='gray', zorder=0)
ax.set_axisbelow(True)

# Límites de ejes
ax.margins(x=0.01)
ax.set_ylim(bottom=-0.08, top=2.3
)
ax.set_xlim(left=df_prep['datetime'].min(), right=df_prep['datetime'].max())

# LEYENDA MEJORADA
legend = ax.legend(loc='upper left', 
                   frameon=True, 
                   fancybox=False,
                   shadow=False, 
                   fontsize=15,
                   framealpha=0.98,
                   edgecolor='black',
                   borderpad=0.8,
                   labelspacing=0.5,
                   handlelength=2.0,
                   handletextpad=0.6)
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_facecolor('white')

# Ajuste fino de las líneas y bordes
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color('black')

plt.tight_layout()


# === 10. GUARDAR CON MEJOR RESOLUCIÓN ===
print("Generando gráficos de alta calidad para Q1...")
plt.savefig('PV_Power_Comparison_Q1.png', 
            dpi=300, 
            bbox_inches='tight', 
            format='png',
            pad_inches=0.12)

plt.savefig('PV_Power_Comparison_Q1.pdf', 
            dpi=300, 
            bbox_inches='tight', 
            format='pdf',
            pad_inches=0.12)

print("✓ Gráficos guardados: 'PV_Power_Comparison_Q1.png' y '.pdf'")
print("✓ Resolución: 300 DPI (estándar Q1)")
print("✓ Formato: Listo para impresión y edición en LaTeX")

plt.show()
