# 🎟️ Optimización de Gastos de Marketing (Cálculo de LTV, CAC y ROMI para Showz)

## 🎯 Breve Descripción del Problema o Propósito

El departamento de analítica de **Showz** (empresa de venta de entradas de eventos) busca optimizar su inversión en publicidad. El objetivo principal de este proyecto es realizar un análisis exhaustivo del comportamiento del cliente y la rentabilidad del marketing para determinar **cuánto dinero invertir y dónde**.

El análisis se centra en responder preguntas críticas de negocio, incluyendo:
1.  **Uso del Servicio:** Frecuencia de visitas, duración de la sesión y recurrencia de usuarios.
2.  **Conversión:** ¿Cuánto tiempo tarda un usuario en realizar su primera compra (registro a conversión)?
3.  **Rentabilidad:** Cálculo de métricas clave como **LTV** (Valor de Vida del Cliente), **CAC** (Costo de Adquisición de Clientes) y **ROMI** (Retorno de la Inversión en Marketing) por fuente de adquisición y a lo largo del tiempo.
4.  **Recomendaciones:** Proponer una estrategia de inversión fundamentada en las fuentes de marketing con mejor ROMI.


---

## 🛠️ Tecnologías Usadas

| Categoría | Herramientas/Librerías | Propósito Específico |
| :--- | :--- | :--- |
| **Lenguaje** | Python | Procesamiento y análisis de datos de visitas, pedidos y costos. |
| **Análisis de Datos** | Pandas | Limpieza de datos, manipulación de *timestamps*, fusión de *datasets* y cálculos de métricas financieras (LTV, CAC, ROMI) mediante análisis de cohortes. |
| **Estadística** | NumPy | Agregación de datos y cálculos estadísticos necesarios para las métricas. |
| **Visualización** | Matplotlib, Seaborn | Creación de gráficos de tendencias para métricas clave (visitas, pedidos, ROMI) a lo largo del tiempo y por fuente de adquisición/dispositivo. |
| **Análisis Financiero** | Análisis de Cohortes | Metodología clave para rastrear el comportamiento del cliente y calcular la rentabilidad a lo largo del tiempo. |
