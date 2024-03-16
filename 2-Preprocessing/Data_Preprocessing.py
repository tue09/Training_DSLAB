import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import random

#df = data frame

def value_count(df):
    for var in df.columns:
        print(df[var].value_counts())
        print("--------------------------------------")

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        mean_=np.mean(subdf.price_per_sqft)
        std_=np.std(subdf.price_per_sqft)
        df_out = pd.concat([df_out, subdf[(subdf.price_per_sqft>(mean_-std_)) & subdf.price_per_sqft<=(mean_+std_)]], ignore_index=True)
    return df_out

def remove_bhk_outliers(df):
    # tạo mảng numpy exlcude_indices để xóa khỏi DataFrame
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'): #Nhóm data theo location
        bhk_stats = {} # tạo một dictionary để lưu thông tin theo bhk
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean':np.mean(bhk_df.price_per_sqft), #Lưu thông tin theo mean
                'std':np.std(bhk_df.price_per_sqft), #Lưu thông tin theo std
                'count':bhk_df.shape[0] #Lưu thông tin theo số mẫu
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk) #Lấy thông tin thống kê của bhk hiện tại
            if stats and stats['count'] > 5: #Nếu dữ liệu > 5 mẫu
                exclude_indices = np.append(exclude_indices, bhk_df[(bhk_df.price_per_sqft<(stats['mean'] - stats['std'])) 
                                | (bhk_df.price_per_sqft>=(stats['mean'] + stats['std']))].index.values)
                # Dữ liệu có giá trị < mean - std hoặc >= mean + std thì sẽ bị vứt vào cái exclude_indices
    return df.drop(exclude_indices, axis='index') #Thực hiện xóa các phần tử

def plot_scatter_chart(df, location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]

    plt.scatter(bhk2.total_sqft_float, bhk2.price, color='Blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft_float, bhk3.price, color='Red', label='3 BHK', s=50, marker='+')
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()


    pass

if __name__ == "__main__":
    path = "Bangalore_House_Price_data/Bengaluru_House_Data.csv"
    df_raw = pd.read_csv(path)
    
    df = df_raw.copy()
    
    df2 = df.drop('society', axis='columns')
    
    df2['balcony'] = df2['balcony'].fillna(df2['balcony'].mean())

    df3 = df2.dropna()

    pd.set_option("display.max_columns", None)
    #pd.set_option("display.max_rows", None)

    
    #print(df3['total_sqft'].value_counts())

    total_sqft_float = []
    for str_val in df3['total_sqft']:
        try:
            total_sqft_float.append(float(str_val))
        except:
            try:
                temp = []
                temp = str_val.split('-')
                total_sqft_float.append((float(temp[0]) + float([temp[-1]]))/2)
            except:
                total_sqft_float.append(np.nan)

    df4 = df3.reset_index(drop=True)

    df5 = df4.join(pd.DataFrame({'total_sqft_float':total_sqft_float}))

    df6 = df5.dropna()

    size_int = []
    for str_val in df6['size']:
        temp=[]
        temp=str_val.split(" ")
        try:
            size_int.append(int(temp[0]))
        except:
            size_int.append(np.nan)
            print("Noise = ", str_val)
    
    df6 = df6.reset_index(drop=True)

    df7 = df6.join(pd.DataFrame({'bhk':size_int}))

    #sns.boxplot(x = df7['total_sqft_float'])

    df_temp7 = df7[df7['total_sqft_float'] < 3500]

    #sns.boxplot(x = df_temp7['total_sqft_float'])

    df8 = df7[(df7['total_sqft_float'] > 500) & (df7['total_sqft_float'] < 2000)]
    #sns.boxplot(x = df8['total_sqft_float'])

    df8 = df8.reset_index(drop=True)
    df8['price_per_sqft'] = df8['price']*100000 / df8['total_sqft_float']

    #Exercise
    #---------------------------
    #Ex0: Sử dụng sns.boxplot() để quan sát đặc điểm 
    #phân bố dữ liệu của các trường số, mỗi trường này 
    #có outlier(ngoại lệ) hay không?

    vars = ['price', 'total_sqft', 'price_per_sqft', 'balcony', 'bath', 'bhk']
    #plt.figure(figsize=(16,12))
    
    #sns.boxplot(x = df8[vars[5]])
    # => moi cai balcony la khong co outliers, con lai la co het

    #---------------------------
    #Ex1: Viết hàm bỏ đi các điểm dữ liệu có price per sqft dựa trên 
    #mean, std của các ngôi nhà dựa trên từng vị trí
    df9 = remove_pps_outliers(df8)

    #---------------------------
    #Ex2: Loại bỏ outlier xét theo trường bhk (số phòng)
    #Xét theo từng khu vực địa lí và theo từng loại nhà với số lượng
    #phòng khác nhau, có một số ngôi nhà có giá không hợp lý (outliers),
    #hãy tìm cách loại bỏ các outliers này. Cần ghi rõ quy tắc ghi nhận outlier
    df10 = remove_bhk_outliers(df9)

    #---------------------------
    #Ex3: Loại bỏ outlier khi xét trường 'bathroom'
    df11 = df10[df10.bath < df10.bhk + 2]

    '''for i, var in enumerate(vars):
        plt.subplot(3, 2, i+1)
        sns.boxplot(df11[var])'''

    #---------------------------
    #Ex4: Xem xét bỏ đi các trường không cần thiết

    df12 = df11.drop(['area_type', 'availability', 'location', 'size', 'total_sqft'], axis=1)

    #Lưu kết quả vào file csv
    df12.to_csv("Bangalore_House_Price_data/clean_data_tue.csv", index=False)

    #---------------------------
    #Ex5: Viết hàm trực quan hóa thể hiện mối tương quan giữa tổng diện tích (total_sqft)
    #và giá nhà (price) theo từng vị trí địa lý (location) (tùy chọn minh họa theo 2 vị trí nào đó).
    #của những căn nhà có 2 hoặc 3 phòng. Và cần phân biệt rõ điểm dữ liệu nào tương ứng với nhà có
    #2 phòng, điểm nào tương ứng nhà có 3 phòng

    #plot_scatter_chart(df9, "Rajaji Nagar")

    #---------------------------
    #Ex6: Thực hiện các câu lệnh để trả lời các câu hỏi dưới đây:
    #a. Thống kê giá nhà theo từng loại khu vực (area_type). Làm với df9
    #b. Xem xét theo từng khu vực, thì giá nhà trung bình (price_per_sqft)
    #là bao nhiêu, tương quan về giá nhà trung bình giữa các khu vực

    df91 = df9.groupby('area_type')['price_per_sqft'].mean().reset_index(name='money')
    df91 = df91.sort_values(by = 'money')
    
    df91['money'] = df91['money'].apply(lambda x : round(x, 2))
    n = df91['area_type'].unique().__len__()+1
    all_colors = list(plt.cm.colors.cnames.keys())

    random.seed(100)
    c = random.choices(all_colors, k=n)
    plt.bar(df91['area_type'], df91['money'], color=c, width=.5)
    for i, val in enumerate(df91['money'].values):
        plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})

    plt.gca().set_xticklabels(df2['area_type'], rotation=60, horizontalalignment= 'right')
    plt.title("Biểu đồ thể hiện giá nhà đất trung bình theo khu vực", fontsize=22)
    plt.ylabel('amount of money')
    plt.show()

    