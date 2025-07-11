import pandas as pd
data = [
    {
        "id_trim": 1,
        "Marca": "BMW",
        "Modelo": "X5",
        "Generacion": "4th Gen (G05)",
        "Ano_desde": 2018.0,
        "Ano_hasta": 2024.0,
        "Serie": "xDrive30d",
        "Acabado": "Luxury Line",
        "Tipo_carroceria": "SUV",
        "altura_carga_mm": 650.0,
        "numero_asientos": 5,
        "longitud_mm": 4922.0,
        "ancho_mm": 2004.0,
        "altura_mm": 1745.0,
        "distancia_ejes_mm": 2975.0,
        "via_delantera_mm": 1670.0,
        "via_trasera_mm": 1670.0,
        "peso_vacio_kg": 2135.0,
        "distancia_ruedas_r14": 2550.0,
        "altura_libre_suelo_mm": 214.0,
        "carga_remolque_frenos_kg": 2700.0,
        "carga_util_kg": 770.0,
        "ancho_via_trasera_mm": 1670.0,
        "ancho_via_delantera_mm": 1670.0,
        "altura_libre_mm": 214.0,
        "peso_total_kg": 2905.0,
        "carga_eje_del_tras_kg": 1450.0,
        "capacidad_max_maletero_l": 1870.0,
        "longitud_ancho_alto_mm": "4922x2004x1745",
        "volumen_carga_m3": 5.1,
        "capacidad_min_maletero_l": 650.0,
        "Par_max_nm": 620.0,
        "Tipo_inyeccion": "Common rail",
        "arbol_levas_cabeza": "DOHC",
        "Disposicion_cilindros": "Inline",
        "Numero_cilindros": 6,
        "Relacion_compresion": 16.5,
        "Tipo_motor": "Diesel",
        "Valvulas_cilindro": 4,
        "Tipo_sobrealimentacion": "Twin-Turbo",
        "Diametro_cilindro_mm": 84.0,
        "Ciclo_carrera_mm": 90.0,
        "Ubicacion_motor": "Front",
        "Diametro_cilindro_ciclo_carrera_mm": "84x90",
        "Rpm_par_maximo": 2000.0,
        "Potencia_max_kW": 210.0,
        "Presencia_intercooler": "Sí",
        "Cilindrada_cm3": 2993.0,
        "CV_motor": 286.0,
        "CV_motor_rpm": 4000,
        "Ruedas_motriz": "All wheel drive",
        "Relacion_diametro-carrera": 0.93,
        "Numero_marchas": 8,
        "Circuito_giro_m": 12.6,
        "Transmision": "Automática",
        "Consumo_combustible_mixto_100_km": 6.6,
        "rango_km": 900,
        "emisiones_estandar": "Euro 6d",
        "capacidad_deposito_combustible_l": 80,
        "aceleracion_0_100_km/h": 6.1,
        "velocidad_max_km_h": 230,
        "combustible_urbano_100_km_l": 7.8,
        "emsiones_c2_g_km": 174.0,
        "calidad_combustible": "Diesel",
        "combustible_carretera_100_km_l": 5.7,
        "suspension_trasera": "Air suspension",
        "freno_trasero": "ventilated disc",
        "freno_delantero": "ventilated disc",
        "suspension_delantera": "Double wishbone",
        "tipo_direccion": "Electric",
        "clase_vehiculo": "SUV Premium",
        "pais_origen": "Alemania",
        "numero_puertas": 5,
        "evaluacion_seguridad": "5 estrellas Euro NCAP",
        "nombre_clasificacion": "SUV Diesel",
        "capacidad_bateria_kw_h": 0.0,
        "rango_electrico_km": 0.0,
        "tiempo_carga_h": 0.0,
    },
    {
        "id_trim": 2,
        "Marca": "Toyota",
        "Modelo": "Corolla",
        "Generacion": "12th Gen",
        "Ano_desde": 2018.0,
        "Ano_hasta": 2023.0,
        "Serie": "Sedan",
        "Acabado": "1.8 VVT-i Hybrid",
        "Tipo_carroceria": "Sedan",
        "altura_carga_mm": 450.0,
        "numero_asientos": 5,
        "longitud_mm": 4630.0,
        "ancho_mm": 1780.0,
        "altura_mm": 1435.0,
        "distancia_ejes_mm": 2700.0,
        "via_delantera_mm": 1530.0,
        "via_trasera_mm": 1530.0,
        "peso_vacio_kg": 1350.0,
        "distancia_ruedas_r14": 2450.0,
        "altura_libre_suelo_mm": 135.0,
        "carga_remolque_frenos_kg": 750.0,
        "carga_util_kg": 450.0,
        "ancho_via_trasera_mm": 1530.0,
        "ancho_via_delantera_mm": 1530.0,
        "altura_libre_mm": 135.0,
        "peso_total_kg": 1800.0,
        "carga_eje_del_tras_kg": 900.0,
        "capacidad_max_maletero_l": 471.0,
        "longitud_ancho_alto_mm": "4630x1780x1435",
        "volumen_carga_m3": 2.8,
        "capacidad_min_maletero_l": 471.0,
        "Par_max_nm": 142.0,
        "Tipo_inyeccion": "Hybrid",
        "arbol_levas_cabeza": "DOHC",
        "Disposicion_cilindros": "Inline",
        "Numero_cilindros": 4,
        "Relacion_compresion": 13.0,
        "Tipo_motor": "Gasoline/Electric",
        "Valvulas_cilindro": 4,
        "Tipo_sobrealimentacion": "None",
        "Diametro_cilindro_mm": 80.5,
        "Ciclo_carrera_mm": 88.3,
        "Ubicacion_motor": "Front",
        "Diametro_cilindro_ciclo_carrera_mm": "80.5x88.3",
        "Rpm_par_maximo": 3600.0,
        "Potencia_max_kW": 90.0,
        "Presencia_intercooler": "No",
        "Cilindrada_cm3": 1798.0,
        "CV_motor": 122.0,
        "CV_motor_rpm": 5200,
        "Ruedas_motriz": "Front wheel drive",
        "Relacion_diametro-carrera": 0.91,
        "Numero_marchas": 1,
        "Circuito_giro_m": 10.8,
        "Transmision": "CVT",
        "Consumo_combustible_mixto_100_km": 3.3,
        "rango_km": 1000,
        "emisiones_estandar": "Euro 6d",
        "capacidad_deposito_combustible_l": 43,
        "aceleracion_0_100_km/h": 10.9,
        "velocidad_max_km_h": 180,
        "combustible_urbano_100_km_l": 3.0,
        "emsiones_c2_g_km": 76.0,
        "calidad_combustible": "95",
        "combustible_carretera_100_km_l": 3.5,
        "suspension_trasera": "Multi-link",
        "freno_trasero": "disc",
        "freno_delantero": "ventilated disc",
        "suspension_delantera": "MacPherson",
        "tipo_direccion": "Electric",
        "clase_vehiculo": "Compacto",
        "pais_origen": "Japón",
        "numero_puertas": 4,
        "evaluacion_seguridad": "5 estrellas Euro NCAP",
        "nombre_clasificacion": "Sedan Híbrido",
        "capacidad_bateria_kw_h": 1.3,
        "rango_electrico_km": 2.0,
        "tiempo_carga_h": 0.0,
    },
    {
        "id_trim": 3,
        "Marca": "Volkswagen",
        "Modelo": "Golf",
        "Generacion": "8th Gen",
        "Ano_desde": 2020.0,
        "Ano_hasta": 2024.0,
        "Serie": "Hatchback",
        "Acabado": "1.5 TSI",
        "Tipo_carroceria": "Hatchback",
        "altura_carga_mm": 670.0,
        "numero_asientos": 5,
        "longitud_mm": 4284.0,
        "ancho_mm": 1789.0,
        "altura_mm": 1456.0,
        "distancia_ejes_mm": 2636.0,
        "via_delantera_mm": 1543.0,
        "via_trasera_mm": 1511.0,
        "peso_vacio_kg": 1295.0,
        "distancia_ruedas_r14": 2450.0,
        "altura_libre_suelo_mm": 142.0,
        "carga_remolque_frenos_kg": 1300.0,
        "carga_util_kg": 535.0,
        "ancho_via_trasera_mm": 1511.0,
        "ancho_via_delantera_mm": 1543.0,
        "altura_libre_mm": 142.0,
        "peso_total_kg": 1830.0,
        "carga_eje_del_tras_kg": 910.0,
        "capacidad_max_maletero_l": 380.0,
        "longitud_ancho_alto_mm": "4284x1789x1456",
        "volumen_carga_m3": 2.2,
        "capacidad_min_maletero_l": 380.0,
        "Par_max_nm": 250.0,
        "Tipo_inyeccion": "Direct injection",
        "arbol_levas_cabeza": "DOHC",
        "Disposicion_cilindros": "Inline",
        "Numero_cilindros": 4,
        "Relacion_compresion": 10.5,
        "Tipo_motor": "Gasoline",
        "Valvulas_cilindro": 4,
        "Tipo_sobrealimentacion": "Turbo",
        "Diametro_cilindro_mm": 74.5,
        "Ciclo_carrera_mm": 85.9,
        "Ubicacion_motor": "Front",
        "Diametro_cilindro_ciclo_carrera_mm": "74.5x85.9",
        "Rpm_par_maximo": 1500.0,
        "Potencia_max_kW": 110.0,
        "Presencia_intercooler": "Sí",
        "Cilindrada_cm3": 1498.0,
        "CV_motor": 150.0,
        "CV_motor_rpm": 5000,
        "Ruedas_motriz": "Front wheel drive",
        "Relacion_diametro-carrera": 0.87,
        "Numero_marchas": 6,
        "Circuito_giro_m": 10.7,
        "Transmision": "Manual",
        "Consumo_combustible_mixto_100_km": 5.2,
        "rango_km": 850,
        "emisiones_estandar": "Euro 6d",
        "capacidad_deposito_combustible_l": 50,
        "aceleracion_0_100_km/h": 8.5,
        "velocidad_max_km_h": 216,
        "combustible_urbano_100_km_l": 6.1,
        "emsiones_c2_g_km": 119.0,
        "calidad_combustible": "95",
        "combustible_carretera_100_km_l": 4.6,
        "suspension_trasera": "Multi-link",
        "freno_trasero": "disc",
        "freno_delantero": "ventilated disc",
        "suspension_delantera": "MacPherson",
        "tipo_direccion": "Electric",
        "clase_vehiculo": "Compacto",
        "pais_origen": "Alemania",
        "numero_puertas": 5,
        "evaluacion_seguridad": "5 estrellas Euro NCAP",
        "nombre_clasificacion": "Hatchback Compacto",
        "capacidad_bateria_kw_h": 0.0,
        "rango_electrico_km": 0.0,
        "tiempo_carga_h": 0.0,
    },
    {
        "id_trim": 4,
        "Marca": "Tesla",
        "Modelo": "Model 3",
        "Generacion": "1st Gen",
        "Ano_desde": 2017.0,
        "Ano_hasta": 2024.0,
        "Serie": "Standard Range Plus",
        "Acabado": "Electric",
        "Tipo_carroceria": "Sedan",
        "altura_carga_mm": 430.0,
        "numero_asientos": 5,
        "longitud_mm": 4694.0,
        "ancho_mm": 1849.0,
        "altura_mm": 1443.0,
        "distancia_ejes_mm": 2875.0,
        "via_delantera_mm": 1580.0,
        "via_trasera_mm": 1570.0,
        "peso_vacio_kg": 1625.0,
        "distancia_ruedas_r14": 2500.0,
        "altura_libre_suelo_mm": 140.0,
        "carga_remolque_frenos_kg": 1000.0,
        "carga_util_kg": 390.0,
        "ancho_via_trasera_mm": 1570.0,
        "ancho_via_delantera_mm": 1580.0,
        "altura_libre_mm": 140.0,
        "peso_total_kg": 2015.0,
        "carga_eje_del_tras_kg": 1010.0,
        "capacidad_max_maletero_l": 542.0,
        "longitud_ancho_alto_mm": "4694x1849x1443",
        "volumen_carga_m3": 3.1,
        "capacidad_min_maletero_l": 425.0,
        "Par_max_nm": 375.0,
        "Tipo_inyeccion": "N/A",
        "arbol_levas_cabeza": "N/A",
        "Disposicion_cilindros": "N/A",
        "Numero_cilindros": 0,
        "Relacion_compresion": 0.0,
        "Tipo_motor": "Electric",
        "Valvulas_cilindro": 0,
        "Tipo_sobrealimentacion": "None",
        "Diametro_cilindro_mm": 0.0,
        "Ciclo_carrera_mm": 0.0,
        "Ubicacion_motor": "Rear",
        "Diametro_cilindro_ciclo_carrera_mm": "N/A",
        "Rpm_par_maximo": 0.0,
        "Potencia_max_kW": 208.0,
        "Presencia_intercooler": "No",
        "Cilindrada_cm3": 0.0,
        "CV_motor": 283.0,
        "CV_motor_rpm": 0,
        "Ruedas_motriz": "Rear wheel drive",
        "Relacion_diametro-carrera": 0.0,
        "Numero_marchas": 1,
        "Circuito_giro_m": 11.6,
        "Transmision": "Automática",
        "Consumo_combustible_mixto_100_km": 0.0,
        "rango_km": 491,
        "emisiones_estandar": "Cero Emisiones",
        "capacidad_deposito_combustible_l": 0,
        "aceleracion_0_100_km/h": 5.6,
        "velocidad_max_km_h": 225,
        "combustible_urbano_100_km_l": 0.0,
        "emsiones_c2_g_km": 0.0,
        "calidad_combustible": "Electricidad",
        "combustible_carretera_100_km_l": 0.0,
        "suspension_trasera": "Multi-link",
        "freno_trasero": "ventilated disc",
        "freno_delantero": "ventilated disc",
        "suspension_delantera": "Double wishbone",
        "tipo_direccion": "Electric",
        "clase_vehiculo": "Eléctrico",
        "pais_origen": "EE.UU.",
        "numero_puertas": 4,
        "evaluacion_seguridad": "5 estrellas Euro NCAP",
        "nombre_clasificacion": "Sedán Eléctrico",
        "capacidad_bateria_kw_h": 60.0,
        "rango_electrico_km": 491.0,
        "tiempo_carga_h": 6.0,
    },
    {
        "id_trim": 5,
        "Marca": "Renault",
        "Modelo": "Kangoo",
        "Generacion": "3rd Gen",
        "Ano_desde": 2021.0,
        "Ano_hasta": 2024.0,
        "Serie": "Van",
        "Acabado": "Blue dCi 95",
        "Tipo_carroceria": "Furgoneta",
        "altura_carga_mm": 570.0,
        "numero_asientos": 2,
        "longitud_mm": 4486.0,
        "ancho_mm": 1860.0,
        "altura_mm": 1860.0,
        "distancia_ejes_mm": 2716.0,
        "via_delantera_mm": 1596.0,
        "via_trasera_mm": 1580.0,
        "peso_vacio_kg": 1425.0,
        "distancia_ruedas_r14": 2450.0,
        "altura_libre_suelo_mm": 158.0,
        "carga_remolque_frenos_kg": 1500.0,
        "carga_util_kg": 850.0,
        "ancho_via_trasera_mm": 1580.0,
        "ancho_via_delantera_mm": 1596.0,
        "altura_libre_mm": 158.0,
        "peso_total_kg": 2275.0,
        "carga_eje_del_tras_kg": 1120.0,
        "capacidad_max_maletero_l": 4000.0,
        "longitud_ancho_alto_mm": "4486x1860x1860",
        "volumen_carga_m3": 3.9,
        "capacidad_min_maletero_l": 3400.0,
        "Par_max_nm": 260.0,
        "Tipo_inyeccion": "Common rail",
        "arbol_levas_cabeza": "DOHC",
        "Disposicion_cilindros": "Inline",
        "Numero_cilindros": 4,
        "Relacion_compresion": 15.6,
        "Tipo_motor": "Diesel",
        "Valvulas_cilindro": 4,
        "Tipo_sobrealimentacion": "Turbo",
        "Diametro_cilindro_mm": 76.0,
        "Ciclo_carrera_mm": 80.5,
        "Ubicacion_motor": "Front",
        "Diametro_cilindro_ciclo_carrera_mm": "76x80.5",
        "Rpm_par_maximo": 1750.0,
        "Potencia_max_kW": 70.0,
        "Presencia_intercooler": "Sí",
        "Cilindrada_cm3": 1461.0,
        "CV_motor": 95.0,
        "CV_motor_rpm": 3750,
        "Ruedas_motriz": "Front wheel drive",
        "Relacion_diametro-carrera": 0.94,
        "Numero_marchas": 6,
        "Circuito_giro_m": 11.2,
        "Transmision": "Manual",
        "Consumo_combustible_mixto_100_km": 5.0,
        "rango_km": 900,
        "emisiones_estandar": "Euro 6d",
        "capacidad_deposito_combustible_l": 54,
        "aceleracion_0_100_km/h": 13.5,
        "velocidad_max_km_h": 161,
        "combustible_urbano_100_km_l": 5.7,
        "emsiones_c2_g_km": 130.0,
        "calidad_combustible": "Diesel",
        "combustible_carretera_100_km_l": 4.5,
        "suspension_trasera": "Leaf spring",
        "freno_trasero": "drum",
        "freno_delantero": "ventilated disc",
        "suspension_delantera": "MacPherson",
        "tipo_direccion": "Electric",
        "clase_vehiculo": "Furgoneta",
        "pais_origen": "Francia",
        "numero_puertas": 4,
        "evaluacion_seguridad": "4 estrellas Euro NCAP",
        "nombre_clasificacion": "Comercial Ligero",
        "capacidad_bateria_kw_h": 0.0,
        "rango_electrico_km": 0.0,
        "tiempo_carga_h": 0.0,
    }
]
df_sample = pd.DataFrame([data])
print(df_sample.T)
df_sample.to_csv("sample_5.csv", index=False)