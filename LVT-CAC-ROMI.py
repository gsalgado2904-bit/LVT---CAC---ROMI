# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Carga de datos
visits = pd.read_csv("/datasets/visits_log_us.csv")
orders = pd.read_csv("/datasets/orders_log_us.csv")
costs = pd.read_csv("/datasets/costs_us.csv")

#Correcion de DFs (Visits)
#Se sustituyen espacios y se agregan guiones bajo en reemplazo, al igual que se pasa todas las columnas a minusculas
visits.columns = visits.columns.str.replace(' ', '_').str.lower()
visits['end_ts'] = pd.to_datetime(visits['end_ts'], format='%Y-%m-%d %H:%M:%S')
visits['start_ts'] = pd.to_datetime(visits['start_ts'], format='%Y-%m-%d %H:%M:%S')
print(visits.head())

#Correcion de DFs (orders)
#Se sustituyen espacios y se agregan guiones bajo en reemplazo, al igual que se pasa todas las columnas a minusculas
orders.columns = orders.columns.str.replace(' ', '_').str.lower()
orders['buy_ts'] = pd.to_datetime(orders['buy_ts'], format='%Y-%m-%d %H:%M:%S')
orders.info()

print(orders.head(20))

#Correcion de DFs (costs)
costs.info()

"""Paso 2.- Informes y metricas

Paso 2.1 - Visitas
"""

#Creacion de series dia, mes y año
visits['session_year']  = visits['start_ts'].dt.isocalendar().year
visits['session_month'] = visits['start_ts'].dt.month
visits['session_week']  = visits['start_ts'].dt.isocalendar().week
visits['session_date'] = visits['start_ts'].dt.date

print(visits.head(15))

"""Paso 2.1.1 - Cuantas personas lo usan cada dia, semana y mes."""

#Creacion de colkumnas de identificadores de cohortes por periodos de tiempo
daily_visits = visits.groupby("session_date").agg({"uid" : "nunique"}).mean()
weekly_visits = visits.groupby("session_week").agg({"uid" : "nunique"}).mean()
monthly_visits = visits.groupby("session_month").agg({"uid" : "nunique"}).mean()

print(f"El promedio de visitas diarias es de: {daily_visits.iloc[0]}")
print(f"El promedio de visitas semanales es de: {weekly_visits.iloc[0]}")
print(f"El promedio de visitas mensuales es de: {monthly_visits.iloc[0]}")

"""Paso 2.1.2 - Cuantas sesiones hay por dia"""

#Se hace groupby para revisar la duracion promedio por dia.

daily_act_sess = visits.groupby("session_date").agg({"uid" : "count"}).mean()
print(f"El promedio de sesiones diarias es de: {daily_act_sess.iloc[0]}")

"""Paso 2.1.3 - Promedio de duracion de sesion

"""

avg_session = (visits["end_ts"] - visits["start_ts"]).mean()
print(avg_session)

"""Paso 2.1.4 - Con que frecuencia regresan los usuarios

sticky factor
DAU/WAU - DAU/MAU

"""

visits_dau = daily_visits.iloc[0]
visits_wau = visits.groupby(["session_year","session_week"]).agg({"uid" : "nunique"}).mean()
sticky_factor = (visits_dau / visits_wau.iloc[0])*100
print(f"El porcentaje en el que regresan los usuarios es de: {round(sticky_factor,2)}%")

"""Paso 2.2.1 - Cuando tardan en comprar despues del primer registro

"""

#Registro de cada usuario
user_registration = visits.groupby(["uid", "device"])["start_ts"].min().reset_index()

#Primera compra de usuario
user_first_purchase = orders.groupby("buy_ts")["uid"].min().reset_index()

#Union de DFs.
cohort = user_registration.merge(user_first_purchase,on="uid")

#Series para la diferencia entre resgistro y compra
cohort["cohort_lifetime"] = (cohort["buy_ts"] - cohort["start_ts"]).dt.days


#creacion de grupos para cohortes
condiciones = [cohort["cohort_lifetime"] == 0,
              (cohort["cohort_lifetime"] > 0) & (cohort["cohort_lifetime"] <= 2),
              (cohort["cohort_lifetime"] > 2) & (cohort["cohort_lifetime"] <= 7),
               cohort["cohort_lifetime"] > 7]

valores = ["Mismo día","1-2 Días","3-7 Días","Más de 7 Días"]
cohort["cohort_group"] = np.select(condlist=condiciones,choicelist=valores,default="Sin compra")

frecuencias = cohort.groupby(["cohort_group", "device"])["uid"].count().reset_index()

sns.barplot(data=frecuencias,x="cohort_group",y="uid",hue="device",palette="viridis")
plt.title('Distribución de Frecuencia por Grupo de Cohorte', fontsize=16)
plt.xlabel('Grupo de Cohorte', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.figure(figsize=(100, 60))

print(frecuencias)

"""Paso 2.2.2 - Cuantos pedidos hacen durante un periodo de tiempo dado


"""

#Se dividen las fechas por dia, mes y año para el buy?ts para sacar promedios de compra por dia.
cohort["buy_year"] = cohort["buy_ts"].dt.isocalendar().year
cohort["buy_month"] = cohort["buy_ts"].dt.month
cohort["buy_week"] = cohort["buy_ts"].dt.isocalendar().week
cohort["buy_date"] = cohort["buy_ts"].dt.date

daily_purchases = cohort.groupby("buy_date")["uid"].count().reset_index()
weekly_purchases = cohort.groupby("buy_week")["uid"].count().reset_index()
monthly_purchases = cohort.groupby("buy_month")["uid"].count().reset_index()

mean_daily_purchase = daily_purchases["uid"].mean()
mean_weekly_purchase = weekly_purchases["uid"].mean()
mean_monthly_purchase = monthly_purchases["uid"].mean()

print(f"El promedio de compras diarios es de: {round(mean_daily_purchase,0)}")
print(f"El promedio de compras semanal es de: {round(mean_weekly_purchase,0)}")
print(f"El promedio de compras mensuales es de: {round(mean_monthly_purchase,0)}")

"""Paso 2.2.3 - Cual es el tamaño promedio de compra"""

mean_revenue = orders["revenue"].mean().round()
print(f"El tamaño promedio de compra es de: ${mean_revenue}")

"""Paso 2.2.4 - Cuanto dinero traen(LTV)"""

#Obtendremos el LTV basandonos en el mes de la primera compra de cada cliente
orders["order_month"] = orders['buy_ts'].astype('datetime64[M]')
costs['month'] = costs['dt'].astype('datetime64[M]')

#recuperamos el mes de la primera compra
first_orders = orders.groupby("uid").agg({"order_month":"min"}).reset_index()
first_orders.columns = ["uid","first_order_month"]

#Calculamos el numero de nuevos clientes cada mes.
cohort_sizes = first_orders.groupby("first_order_month").agg({"uid":"nunique"}).reset_index()
cohort_sizes.columns = ['first_order_month', 'n_buyers']

#combinamos los meses de la primera compra a la tabla orders
orders = pd.merge(orders,first_orders,on="uid")

cohorts_sales = orders.groupby(['first_order_month','order_month']).agg({'revenue': 'sum'}).reset_index()

#Combinamos los cohort_sizes con los cohorts_sales
report = pd.merge(cohort_sizes,cohorts_sales, on="first_order_month")

margin_rate = 0.5

report["gp"] = report["revenue"] * margin_rate
report['age'] = (report['order_month'] - report['first_order_month']) / np.timedelta64(1, 'M')
report['age'] = report['age'].round().astype('int')

print(report)

report['ltv'] = report['gp'] / report['n_buyers']

output = report.pivot_table(
    index='first_order_month',
    columns='age',
    values='ltv',
    aggfunc='mean').round()

output.fillna('')

print(output)

"""2.3 Marketing

2.3.1 - ¿Cuanto dinero se gasto?
"""



print(costs)

#Dinero gastado en total

spent = costs["costs"].sum()
print(f"El gasto total de Marketing es de: ${spent}")

#gastos por fuente de adquisicion

spent_source_id = costs.groupby("source_id")["costs"].sum().reset_index()


sns.barplot(x="source_id",y="costs",data=spent_source_id,palette="viridis")

plt.title('Gastos por fuente de adquisicion', fontsize=16)
plt.xlabel('Fuente de adquisicion', fontsize=12)
plt.ylabel('Costo total', fontsize=12)
plt.figure(figsize=(100, 60))

#Gastos a lo largo del tiempo

spent_monthly = costs.groupby(["month","source_id"])["costs"].sum().reset_index()
spent_monthly["month"] = spent_monthly["month"].dt.date


sns.barplot(x="month",y="costs",data=spent_monthly,hue="source_id")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.title('Gastos por mes', fontsize=16)
plt.xlabel('Mes', fontsize=12)
plt.ylabel('Costo total', fontsize=12)
plt.figure(figsize=(100, 60))

costs.info()

"""2.3.2 Costo de adquisicion de clientes

Costos de la fuente / revenue de nuevos usuarioseports
"""

#aggrupamos las clientes nuevos por mes


nbuyer_per_month = report.groupby("order_month")["n_buyers"].last().reset_index()
nbuyer_per_month.columns = ["month","n_buyers"]


print(nbuyer_per_month)

#Agrupamos los costos por mes y por source_id
costs_per_month = costs.groupby(["month","source_id"])["costs"].sum().reset_index()
print(costs_per_month)

#Separamos el LVT del archivo reports para tenerlo por cohorte mensual

ltv_monthly = report.groupby("order_month")["ltv"].sum().reset_index()
ltv_monthly.columns = ["month","ltv"]
print(ltv_monthly)

#Unimos los dos  3 DFs, para poder calcular el CAC en cada uno de los meses.Ademas de facilitarl
report_monthly = pd.merge(nbuyer_per_month,costs_per_month,on="month")
print(costs_revenue)

#unimos el 3er DF para poder obtener todos nuestros datos mensuales en uno mismo.

report_monthly = pd.merge(report_monthly,ltv_monthly,on="month")

#Con los datos ya en un mismo dataframe podemos obtener el CAC por mes,

report_monthly["cac"] = report_monthly["costs"] / report_monthly["n_buyers"]
#Ultima correccion de la columna month, para poder mostrarla de manera correcta en los repores.
report_monthly["month"] = pd.to_datetime(report_monthly["month"])
report_monthly["month"] = report_monthly["month"].dt.strftime("%Y-%m")


report_monthly.info()

"""2.3.3 ROMI"""

report_monthly["romi"] = (report_monthly["ltv"] / report_monthly["cac"])

print(report_monthly)

"""Trazado de graficos y visualizaciones"""

sns.barplot(x="month",y="cac",data=report_monthly,hue="source_id")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.title('CAC por fuente y mes', fontsize=16)
plt.xlabel('Mes', fontsize=12)
plt.ylabel('CAC', fontsize=12)
plt.figure(figsize=(100, 60))
print(report_monthly)

sns.barplot(x="month",y="romi",data=report_monthly,hue="source_id")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.title('ROMI por fuente y mes', fontsize=16)
plt.xlabel('Mes', fontsize=12)
plt.ylabel('ROMI', fontsize=12)
plt.figure(figsize=(100, 60))

"""Conclusiones

Dadas las 3 graficas proporcionadas de: ROMI por fuente y mes, CAC por fuente y mes y gastos por mes por fuente.

La fuente que mejores estadisticas tiene en general es la fuente 10 y 9, ya que estas en conjunto representan una inversion minima comparada con las otras fuentes y son las que mejor se desempeñan tanto en el analisis de CAC y ROMI.

Con estas dos fuentes obtendremos el menor costo de adquisicion de clientes al igual que el mejor retorno de inversion, entonces lo ideal seria recortar costos a los source id 3 y 4 y reinvertirlos en el 9 y 10 ya que 2 y 4 son los que peores inversiones regresan tanto en CAC como en ROMI de esta manera podemos decrementar de manera exponencial el CAC al igual que recuperar la inversion de una manera mas estable sin tantos riesgos
"""
