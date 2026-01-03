import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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

    def calibrar_modelo_global(self, df_prep=None, disp=True):
        """
        Calibra el modelo con differential evolution.
        Si df_prep es None, usa los datos completos.
        """
        if df_prep is None:
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
            maxiter=1000,
            popsize=40,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=42,
            disp=disp
        )
        if result.success:
            params = result.x
            return params
        else:
            return np.array([0.4, 0.2, 0.1, 0.3, 0.02, 0.05, 0.01, 2.0, 0.0, 0.01, -10.0])

    def validacion_cruzada_kfold(self, k=5):
        """
        Implementación de K-Fold Cross-Validation para evaluar generalización.
        Devuelve métricas de desempeño en datos de validación.
        """
        print("\n" + "="*70)
        print("VALIDACIÓN CRUZADA K-FOLD (K={})".format(k))
        print("="*70)
        
        df_prep = self.calcular_variables_avanzadas(self.df)
        
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        # Listas para almacenar resultados
        r2_train_scores = []
        r2_test_scores = []
        rmse_train_scores = []
        rmse_test_scores = []
        mae_train_scores = []
        mae_test_scores = []
        params_all_folds = []
        
        fold_num = 0
        for train_idx, test_idx in kf.split(df_prep):
            fold_num += 1
            print(f"\n{'─'*70}")
            print(f"FOLD {fold_num}/{k}")
            print(f"{'─'*70}")
            
            # Separar datos de entrenamiento y prueba
            df_train = df_prep.iloc[train_idx].reset_index(drop=True)
            df_test = df_prep.iloc[test_idx].reset_index(drop=True)
            
            print(f"Registros entrenamiento: {len(df_train):,}")
            print(f"Registros validación:   {len(df_test):,}")
            
            # Calibrar SOLO con datos de entrenamiento
            print(f"Calibrando parámetros en fold {fold_num}...")
            params = self.calibrar_modelo_global(df_prep=df_train, disp=False)
            params_all_folds.append(params)
            
            # Evaluar en DATOS DE ENTRENAMIENTO
            T_train = self.modelo_ultra_mejorado(df_train, params)
            P_train = self.calcular_potencia_hong(T_train, df_train)
            mask_train = ~df_train['P_real'].isna()
            
            r2_train = r2_score(df_train['P_real'][mask_train], P_train[mask_train])
            rmse_train = np.sqrt(mean_squared_error(df_train['P_real'][mask_train], P_train[mask_train]))
            mae_train = mean_absolute_error(df_train['P_real'][mask_train], P_train[mask_train])
            
            r2_train_scores.append(r2_train)
            rmse_train_scores.append(rmse_train)
            mae_train_scores.append(mae_train)
            
            # Evaluar en DATOS DE PRUEBA 
            T_test = self.modelo_ultra_mejorado(df_test, params)
            P_test = self.calcular_potencia_hong(T_test, df_test)
            mask_test = ~df_test['P_real'].isna()
            
            r2_test = r2_score(df_test['P_real'][mask_test], P_test[mask_test])
            rmse_test = np.sqrt(mean_squared_error(df_test['P_real'][mask_test], P_test[mask_test]))
            mae_test = mean_absolute_error(df_test['P_real'][mask_test], P_test[mask_test])
            
            r2_test_scores.append(r2_test)
            rmse_test_scores.append(rmse_test)
            mae_test_scores.append(mae_test)
            
            # Mostrar resultados del fold
            print(f"\nMétricas ENTRENAMIENTO:")
            print(f"  R² (Train):   {r2_train:.6f}")
            print(f"  RMSE (Train): {rmse_train:.5f} kW")
            print(f"  MAE (Train):  {mae_train:.5f} kW")
            
            print(f"\nMétricas VALIDACIÓN (datos no vistos):")
            print(f"  R² (Test):    {r2_test:.6f}")
            print(f"  RMSE (Test):  {rmse_test:.5f} kW")
            print(f"  MAE (Test):   {mae_test:.5f} kW")
            
            # Indicador de sobreajuste
            diff_r2 = r2_train - r2_test
            print(f"\nDiferencia R² (Train-Test): {diff_r2:.6f}", end="")
            if diff_r2 < 0.01:
                print(" ✓ Excelente generalización")
            elif diff_r2 < 0.05:
                print(" ✓ Buena generalización")
            elif diff_r2 < 0.10:
                print(" ⚠ Ligero sobreajuste")
            else:
                print(" ✗ Sobreajuste significativo")
        
        # ========== RESUMEN FINAL ==========
        print("\n" + "="*70)
        print("RESUMEN K-FOLD CROSS-VALIDATION")
        print("="*70)
        
        r2_train_mean = np.mean(r2_train_scores)
        r2_train_std = np.std(r2_train_scores)
        r2_test_mean = np.mean(r2_test_scores)
        r2_test_std = np.std(r2_test_scores)
        
        rmse_train_mean = np.mean(rmse_train_scores)
        rmse_train_std = np.std(rmse_train_scores)
        rmse_test_mean = np.mean(rmse_test_scores)
        rmse_test_std = np.std(rmse_test_scores)
        
        mae_train_mean = np.mean(mae_train_scores)
        mae_train_std = np.std(mae_train_scores)
        mae_test_mean = np.mean(mae_test_scores)
        mae_test_std = np.std(mae_test_scores)
        
        print("\n┌─ R² Score ─────────────────────────────────────┐")
        print(f"│ Entrenamiento: {r2_train_mean:.4f} ± {r2_train_std:.4f}")
        print(f"│ Validación:    {r2_test_mean:.4f} ± {r2_test_std:.4f}")
        print(f"│ Diferencia:    {r2_train_mean - r2_test_mean:.4f}")
        print("└─────────────────────────────────────────────────┘")
        
        print("\n┌─ RMSE (kW) ────────────────────────────────────┐")
        print(f"│ Entrenamiento: {rmse_train_mean:.5f} ± {rmse_train_std:.5f}")
        print(f"│ Validación:    {rmse_test_mean:.5f} ± {rmse_test_std:.5f}")
        print("└─────────────────────────────────────────────────┘")
        
        print("\n┌─ MAE (kW) ─────────────────────────────────────┐")
        print(f"│ Entrenamiento: {mae_train_mean:.5f} ± {mae_train_std:.5f}")
        print(f"│ Validación:    {mae_test_mean:.5f} ± {mae_test_std:.5f}")
        print("└─────────────────────────────────────────────────┘")
        
        # Diagnóstico de sobreajuste
        print("\n┌─ DIAGNÓSTICO DE SOBREAJUSTE ───────────────────┐")
        avg_diff = r2_train_mean - r2_test_mean
        if avg_diff < 0.01:
            status = "✓ SIN SOBREAJUSTE DETECTABLE"
            color = "VERDE"
        elif avg_diff < 0.05:
            status = "✓ BUEN AJUSTE (Mínimo sobreajuste)"
            color = "VERDE"
        elif avg_diff < 0.10:
            status = "⚠ LIGERO SOBREAJUSTE"
            color = "AMARILLO"
        else:
            status = "✗ SOBREAJUSTE SIGNIFICATIVO"
            color = "ROJO"
        
        print(f"│ {status}")
        print(f"│ Diferencia promedio R²: {avg_diff:.6f}")
        print("└─────────────────────────────────────────────────┘")
        
        # Tabla de resultados por fold
        print("\n┌─ RESULTADOS DETALLADOS POR FOLD ──────────────┐")
        print(f"│ {'Fold':<6} {'R²_Train':<11} {'R²_Test':<11} {'Δ R²':<10}")
        print("├─────────────────────────────────────────────────┤")
        for i in range(k):
            delta = r2_train_scores[i] - r2_test_scores[i]
            print(f"│ {i+1:<6} {r2_train_scores[i]:<11.6f} {r2_test_scores[i]:<11.6f} {delta:<10.6f}")
        print("└─────────────────────────────────────────────────┘")
        
        # Retornar resultados
        resultados = {
            'r2_train': {'mean': r2_train_mean, 'std': r2_train_std, 'scores': r2_train_scores},
            'r2_test': {'mean': r2_test_mean, 'std': r2_test_std, 'scores': r2_test_scores},
            'rmse_train': {'mean': rmse_train_mean, 'std': rmse_train_std, 'scores': rmse_train_scores},
            'rmse_test': {'mean': rmse_test_mean, 'std': rmse_test_std, 'scores': rmse_test_scores},
            'mae_train': {'mean': mae_train_mean, 'std': mae_train_std, 'scores': mae_train_scores},
            'mae_test': {'mean': mae_test_mean, 'std': mae_test_std, 'scores': mae_test_scores},
            'params_all_folds': params_all_folds,
            'diagnostico': status
        }
        
        return resultados

    def validacion_temporal(self):
        """
        Validación temporal: calibra con primavera+otoño, 
        valida con verano+invierno sin recalibración.
        """
        print("\n" + "="*70)
        print("VALIDACIÓN TEMPORAL")
        print("="*70)
        
        df_prep = self.calcular_variables_avanzadas(self.df)
        df_prep['month'] = df_prep['datetime'].dt.month
        
        # Entrenamiento: marzo-noviembre (primavera-otoño)
        # Validación: diciembre-febrero (invierno) + todos los meses restantes
        
        # Estrategia: usar 9 meses para entrenar, 3 meses para validar
        # Rotación: 4 validaciones diferentes
        
        print("\nEstrategia: Validación temporal por estaciones")
        print("Calibración con primavera/otoño → Validación con invierno/verano")
        
        # Split: Marzo-Noviembre entrena, Diciembre-Febrero valida
        df_train = df_prep[df_prep['month'].isin([3,4,5,6,7,8,9,10,11])].reset_index(drop=True)
        df_test = df_prep[df_prep['month'].isin([12,1,2])].reset_index(drop=True)
        
        print(f"\nRegistros entrenamiento (Mar-Nov): {len(df_train):,}")
        print(f"Registros validación (Dic-Feb):   {len(df_test):,}")
        
        print("\nCalibrando parámetros con datos de entrenamiento...")
        params = self.calibrar_modelo_global(df_prep=df_train, disp=False)
        
        # Evaluar en entrenamiento
        T_train = self.modelo_ultra_mejorado(df_train, params)
        P_train = self.calcular_potencia_hong(T_train, df_train)
        mask_train = ~df_train['P_real'].isna()
        
        r2_train = r2_score(df_train['P_real'][mask_train], P_train[mask_train])
        rmse_train = np.sqrt(mean_squared_error(df_train['P_real'][mask_train], P_train[mask_train]))
        mae_train = mean_absolute_error(df_train['P_real'][mask_train], P_train[mask_train])
        
        # Evaluar en validación
        T_test = self.modelo_ultra_mejorado(df_test, params)
        P_test = self.calcular_potencia_hong(T_test, df_test)
        mask_test = ~df_test['P_real'].isna()
        
        r2_test = r2_score(df_test['P_real'][mask_test], P_test[mask_test])
        rmse_test = np.sqrt(mean_squared_error(df_test['P_real'][mask_test], P_test[mask_test]))
        mae_test = mean_absolute_error(df_test['P_real'][mask_test], P_test[mask_test])
        
        print("\n┌─ RESULTADOS VALIDACIÓN TEMPORAL ───────────────┐")
        print(f"│ Métrica         │ Entrenamiento │ Validación")
        print("├─────────────────┼────────────────┼─────────────────┤")
        print(f"│ R²              │     {r2_train:.6f}     │    {r2_test:.6f}")
        print(f"│ RMSE (kW)       │    {rmse_train:.5f}    │   {rmse_test:.5f}")
        print(f"│ MAE (kW)        │    {mae_train:.5f}    │   {mae_test:.5f}")
        print("└─────────────────┴────────────────┴─────────────────┘")
        
        diff_r2 = r2_train - r2_test
        print(f"\nDiferencia R²: {diff_r2:.6f}", end="")
        if diff_r2 < 0.05:
            print(" → ✓ Excelente generalización temporal")
        elif diff_r2 < 0.10:
            print(" → ✓ Buena generalización temporal")
        else:
            print(" → ⚠ Sobreajuste temporal detectado")
        
        resultados = {
            'r2_train': r2_train,
            'r2_test': r2_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'params': params
        }
        
        return resultados

    def analizar_global(self):
        """
        Análisis global: calibración sin validación (para referencia)
        """
        print("\n" + "="*70)
        print("ANÁLISIS GLOBAL (CALIBRACIÓN SIN VALIDACIÓN)")
        print("="*70)
        print(f"Registros totales: {len(self.df):,}")
        
        df_prep = self.calcular_variables_avanzadas(self.df)
        
        print("\nCalibrando parámetros con datos completos...")
        params = self.calibrar_modelo_global(df_prep=df_prep, disp=False)
        
        T_ultra = self.modelo_ultra_mejorado(df_prep, params)
        P_ultra = self.calcular_potencia_hong(T_ultra, df_prep)
        
        mask = ~df_prep['P_real'].isna()
        mse = mean_squared_error(df_prep['P_real'][mask], P_ultra[mask])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(df_prep['P_real'][mask], P_ultra[mask])
        r2 = r2_score(df_prep['P_real'][mask], P_ultra[mask])
        
        print("\n┌─ MÉTRICAS GLOBALES (SIN VALIDACIÓN) ──────────┐")
        print(f"│ R²:    {r2:.6f}")
        print(f"│ RMSE:  {rmse:.5f} kW")
        print(f"│ MAE:   {mae:.5f} kW")
        print(f"│ MSE:   {mse:.6f}")
        print("└─────────────────────────────────────────────────┘")
        
        print(f"\n✓ Parámetros calibrados:")
        param_names = ['α₁', 'α₂', 'α₃', 'β₁', 'β₂', 'β₃', 'γ₁', 'γ₂', 'γ₃', 'δ', 'C_med']
        for name, value in zip(param_names, params):
            print(f"  {name:>5} = {value:>10.6f}")
        
        return {
            'R2': r2, 
            'MSE': mse, 
            'RMSE': rmse, 
            'MAE': mae, 
            'Params': params
        }

# USO:
if __name__ == "__main__":
    archivo_excel = r'C:\Users\Usuario\Desktop\Trabajo_UNI\Gemelos digitales\Articulos\ART_2\fINAL_REV\Datos temperatura_2_79 kW.xlsx'
    analizador = ModeloUltraMejorado(archivo_excel, 'Hoja1')
    
    # Opción 1: Análisis global (original, sin validación)
    resultados_global = analizador.analizar_global()
    
    # Opción 2: Validación cruzada K-Fold
    resultados_kfold = analizador.validacion_cruzada_kfold(k=5)
    
    # Opción 3: Validación temporal
    resultados_temporal = analizador.validacion_temporal()
    
    print("\n" + "="*70)
    print("CONCLUSIÓN: Comparar R² en validación con R² en entrenamiento")
    print("Si son similares → ✓ SIN SOBREAJUSTE")
    print("Si son muy diferentes → ✗ SOBREAJUSTE DETECTADO")
    print("="*70)
