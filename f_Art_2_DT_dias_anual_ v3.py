import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModeloUltraMejorado:
    def __init__(self, excel_file, sheet_name='Hoja1'):
        self.excel_file = excel_file
        self.sheet_name = sheet_name
        self.cargar_datos()
        
    def cargar_datos(self):
        self.df = pd.read_excel(self.excel_file, sheet_name=self.sheet_name)
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
            if old_name in self.df.columns:
                self.df[new_name] = self.df[old_name]
        # Elimina solo valores nulos de viento, pero conserva los ceros
        self.df = self.df.dropna(subset=['T_amb', 'G', 'w', 'P_real'])
        if 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.NOCT = self.df['NOCT'].iloc[0] if 'NOCT' in self.df.columns else 45.0
        self.T_ref = self.df['T_ref'].iloc[0] if 'T_ref' in self.df.columns else 25.0
        self.P_ref = self.df['P_ref'].iloc[0] if 'P_ref' in self.df.columns else 465.0

    def calcular_variables_avanzadas(self, df):
        df = df.copy()
        df['dG'] = df['G'].diff().fillna(0)
        df['d2G'] = df['dG'].diff().fillna(0)
        df['sigma_G_5'] = df['G'].rolling(window=5, min_periods=1).std().fillna(0)
        df['cloud_indicator'] = ((df['dG'].abs() > 50) & (df['sigma_G_5'] > 100)).astype(float)
        df['w_squared'] = df['w'] ** 2
        df['G_sqrt'] = np.sqrt(df['G'].clip(lower=0))
        return df

    def modelo_ultra_mejorado(self, df, params):
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

        # Límites realistas para Badajoz
        T_MIN, T_MAX = -10, 80      # Temperatura ambiente y celda
        INERCIA_MIN, INERCIA_MAX = -50, 50  # Inercia térmica

        for i in range(n):
            T_amb_i = np.clip(T_amb[i], T_MIN, T_MAX)
            base = T_amb_i + (self.NOCT - 20)/800 * G[i]
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
            viento = (beta1 * (T_amb_i - self.T_ref) * w[i] +
                      beta2 * dG[i] * w[i] +
                      beta3 * w_sq[i] * (T_amb_i - self.T_ref))
            nubosidad = (gamma1 * abs(dG[i]) +
                         gamma2 * sigma_G_5[i] * cloud_ind[i] +
                         gamma3 * d2G[i] * cloud_ind[i])
            no_lineal = delta * G_sqrt[i] * (T_amb_i - self.T_ref)
            T_cell[i] = base + inercia + viento + nubosidad + no_lineal + C_med
            T_cell[i] = np.clip(T_cell[i], T_MIN, T_MAX)
        return T_cell

    def calcular_potencia_hong(self, T_cell, df):
        beta_temp = 0.004
        num_paneles = 6
        power = (self.P_ref * (df['G'] / 1000) * 
                (1 - beta_temp * (T_cell - self.T_ref)) * num_paneles)
        return power / 1000

    def objetivo_multicriterio(self, params, df):
        try:
            T_cell = self.modelo_ultra_mejorado(df, params)
            P_calc = self.calcular_potencia_hong(T_cell, df)
            mask = ~df['P_real'].isna()
            if mask.sum() < 10: return 1e6
            r2 = r2_score(df['P_real'][mask], P_calc[mask])
            return -r2  # Maximizar R2
        except Exception:
            return 1e6

    def calibrar_modelo_global(self):
        df_prep = self.calcular_variables_avanzadas(self.df)
        bounds = [
            (0.0, 0.9),    # α1
            (0.0, 0.8),    # α2
            (0.0, 0.5),    # α3
            (-1.5, 1.5),   # β1
            (-0.2, 0.2),   # β2
            (0.0, 0.2),    # β3
            (0.0, 0.1),    # γ1
            (0.0, 10.0),   # γ2
            (-0.1, 0.1),   # γ3
            (0.0, 0.05),   # δ
            (-25.0, 25.0)  # C_med
        ]
        result = differential_evolution(
            lambda p: self.objetivo_multicriterio(p, df_prep),
            bounds,
            strategy='best1bin',
            maxiter=1000,  # Más iteraciones para mejor ajuste global
            popsize=40,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=42,
            disp=True      # Muestra progreso y warnings
        )
        if result.success:
            params = result.x
            return params, df_prep
        else:
            return [0.4, 0.2, 0.1, 0.3, 0.02, 0.05, 0.01, 2.0, 0.0, 0.01, -10.0], df_prep

    def analizar_global(self):
        print("\n=== ANÁLISIS GLOBAL ===")
        print(f"Registros: {len(self.df)}")
        params, df_prep = self.calibrar_modelo_global()
        T_ultra = self.modelo_ultra_mejorado(df_prep, params)
        P_ultra = self.calcular_potencia_hong(T_ultra, df_prep)
        mask = ~df_prep['P_real'].isna()
        mse = mean_squared_error(df_prep['P_real'][mask], P_ultra[mask])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(df_prep['P_real'][mask], P_ultra[mask])
        r2 = r2_score(df_prep['P_real'][mask], P_ultra[mask])
        print(f"MSE:   {mse:.6f}")
        print(f"RMSE:  {rmse:.5f}")
        print(f"MAE:   {mae:.5f}")
        print(f"R²:    {r2:.4f}")
        print(f"Parámetros calibrados: {params}")
        return {'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'Params': params}

# USO:
if __name__ == "__main__":
    archivo_excel = r'C:\Users\Usuario\Desktop\Trabajo_UNI\Gemelos digitales\Articulos\ART_2\fINAL_REV\Datos temperatura_2_79 kW.xlsx'
    analizador = ModeloUltraMejorado(archivo_excel, 'Hoja1')
    analizador.analizar_global()
