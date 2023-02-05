**İŞ PROBLEMİ**

> FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
> Şirketinortauzunvadeliplan yapabilmesi için var olan müşterilerin gelecekte şirket esağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir

**Veri seti Hikayesi:**
> Flo’danson alışverişlerini 2020 -2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmişalışveriş davranışlarından elde edilen bilgiler deno luşmaktadır.

1. master_id : Eşsiz müşteri numarası
2. order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı
3. last_order_channel : En son alışveriş yaptığı kanal
4. first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
5. last_order_channel : Müşterinin yaptığı son alışveriş tarihi
6. last_order_date_offline : Müşterinin offline platformda yaptığı ilk alışveriş tarihi
7. order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
8. order_num_total_ever_offline : Müşterinin offline platformda yaptığı toplam alışveriş sayısı
9. customer_value_total_ever_offline : Müşterinin offline platformda yaptığı alışverişin toplam ücreti
10. customer_value_total_ever_online : Müşterinin online platformda yaptığı alışverişin toplam ücreti
11. interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import squarify

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#Adım 1: flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz
df_ = pd.read_csv('../input/flo-data-20k/flo_data_20k.csv')
df = df_.copy()


#Adım 2: Veri setindeİlk 10 gözlem
df.head(10)
#Değişken isimleri
df.describe().T
#Betimsel istatistik
df.shape
df.nunique()# essiz urun sayisi nedir?

df.value_counts()#hangi üründen kaçar tane var

df.nunique()
#Boş değer,
df.isnull().sum()

#Değişken tipleri, incelemesi yapınız
df.dtypes

#Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam alışveriş sayısı ve
# harcaması için yeni değişkenler oluşturunuz

df['total_of_purchases'] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df['total_of_pspending'] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.info



# 3. Veri Hazırlama (Data Preparation)
###############################################################
#Veri ön hazırlık sürecini fonksiyonlaştırınız.
def pre_processing (dataframe):
    df.isnull().sum()
    df.dropna(inplace=True)
    df.describe().T
    # order channel , total of purchase and total expenditure distribution
    df.groupby("order_channel").agg({"total_of_purchases": "sum",
                                     "total_of_pspending": "count"}).sort_values(by="total_of_pspending",ascending=False).head(10)
    # En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
    df.groupby("master_id").agg({"total_of_pspending": sum}).sort_values(by="total_of_pspending", ascending=False).head(10)
    # En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
    df.groupby("master_id").agg({"total_of_purchases": "count"}).sort_values(by="total_of_purchases",ascending=False).head(10)

    datetime=["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    df[datetime] = df[datetime].apply(pd.to_datetime)

    return df
pre_processing(df)


# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
###############################################################
# Recency, Frequency, Monetary
df.head()

orderdate=df["last_order_date"].max()
today_date = dt.datetime(2021,7,1)
type(today_date)

rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                   'total_of_purchases': lambda total_of_purchases: total_of_purchases.sum(),
                                   'total_of_pspending': lambda total_of_pspending: total_of_pspending.sum()})

rfm.head()

rfm.columns = ["recency","frequency","monetary"]

rfm.describe().T
rfm.shape



# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
###############################################################
rfm["recency_score"] = pd.qcut(rfm["recency"].rank(method="first"),
                               5, labels=[5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"),
                               5, labels=[5,4,3,2,1])
rfm["monetary_score"] = pd.qcut(rfm["monetary"],
                               5, labels=[5,4,3,2,1])

rfm["RFM_SCORE"]=(rfm["recency_score"].astype(str)+rfm["frequency_score"].astype(str)+rfm["monetary_score"].astype(str))

rfm[rfm["RFM_SCORE"]=="545"]
rfm[rfm["RFM_SCORE"] =="122"]



# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
###############################################################
# regex
# RFM isimlendirmesi
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map,regex=True)
rfm[["segment","recency","frequency","monetary"]].groupby("segment").agg(["mean","count"])

segmentsx = rfm["segment"].value_counts().sort_values(ascending=False)
segmentsx




fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(15, 11)

squarify.plot(sizes=segmentsx,
              label=['hibernating',
                     'at_Risk',
                     'cant_loose',
                     'about_to_sleep',
                     'need_attention',
                     'loyal_customers',
                     'promising',
                     'new_customers',
                     'potential_loyalists',
                     'champions'],color=["red","yellow","blue", "green","orange"],pad=True,
              bar_kwargs={'alpha':.70}, text_kwargs={'fontsize':12})
plt.title("FLO Customer Segmentation",fontsize=20)
plt.xlabel('Frequency', fontsize=20)
plt.ylabel('Recency', fontsize=20)
plt.show()


SEG_1 = rfm[(rfm["segment"]=="champions4") | (rfm["segment"]=="loyal_customers1")]
SEG_1.shape[0]

SEG_2 = df[(df["interested_in_categories_12"]).str.contains("KADIN")] #7603
SEG_2 .shape[0]


cmd_c = pd.merge(SEG_1,SEG_2[["interested_in_categories_12","master_id"]],on=["master_id"])
cmd_c.columns

cmd_c.to_csv("new_customer.csv")

#CASE 2:
SEG_3 = rfm[(rfm["segment"]=="cant_loose1") | (rfm["segment"]=="about_to_sleep1") | (rfm["segment"]=="new_customers1")]
SEG_3.shape[0]

SEG_4= df[(df["interested_in_categories_12"]).str.contains("ERKEK|COCUK")]

cmd_c2 = pd.merge(SEG_3,SEG_4[["interested_in_categories_12","master_id"]],on=["master_id"])
cmd_c2.to_csv("new_customer2.csv")