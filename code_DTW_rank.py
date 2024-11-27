import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from dtaidistance import dtw
import matplotlib.pyplot as plt


def main():
    df = read_csv('/home/jhoh/DL/owid-covid-data_231216_final.csv')

    targets = ["KOR", "JPN", "RUS", "ITA", "USA"]

    df["Date"] = pd.to_datetime(df["Date"])

    start = pd.to_datetime("2022-01-01")
    end = pd.to_datetime("2022-07-31")

    df = df[df["population"] >= 10000000]
    df = df[df["Date"] >= start]
    df = df[df["Date"] <= end]

    iso_codes = df['iso_code'].unique()
    print(len(iso_codes))
    rank = pd.DataFrame(np.zeros([len(iso_codes), len(targets)]))
    distance = pd.DataFrame(np.zeros([len(iso_codes), len(targets)]))
    rank.columns = targets
    distance.columns = targets

    for target in targets:
        df1 = df[[target in c for c in list(df['iso_code'])]]
        df1 = df1[["Date", "new_cases_corrected"]]

        scaler = MinMaxScaler()

        for iso_code in iso_codes[iso_codes != target]:
            df2 = df[[iso_code in c for c in list(df['iso_code'])]]
            df2 = df2[["Date", "new_cases_corrected"]]
            df2.columns = ["Date", iso_code]
            df1 = pd.merge(df1, df2, on="Date", how="outer")

        all = np.insert(iso_codes[iso_codes != target], 0, target)

        df1 = df1.drop("Date", axis=1)
        df1 = df1.replace(0, np.NaN)
        df1 = df1.interpolate(limit_direction="both")
        df1.columns = np.insert(iso_codes[iso_codes != target], 0, target)
        df1 = np.transpose(df1)
        df1_reshape = df1.values.reshape(df1.shape[0], df1.shape[1])
        df1_scaled = scaler.fit_transform(df1_reshape)

        dtw_dist = dtw.distance_matrix_fast(df1_reshape)
        dtw_dist_scaled = dtw.distance_matrix_fast(df1_scaled)
        df_dtw = pd.DataFrame(dtw_dist, index=all, columns=all)
        df_dtw_scaled = pd.DataFrame(dtw_dist_scaled, index=all, columns=all)

        df_dtw_scaled_sorted = df_dtw_scaled.sort_values(by=df_dtw_scaled.columns[0])
        print(df_dtw_scaled_sorted[target].index)

        x_lab = [y + z for y, z in zip(df_dtw_scaled_sorted[target].index,
                                       ["_" + str(w) for w in range(len(df_dtw_scaled_sorted[target].index))])]
        rank[target] = df_dtw_scaled_sorted[target].index
        distance[target] = np.array(df_dtw_scaled_sorted[target])

        plt.figure(figsize=(10, 6))
        plt.plot(x_lab, distance[target])
        plt.title("DTW distance with " + target, fontsize=20)
        plt.xlabel("country_rank of DTW distance")
        plt.ylabel("distance")
        plt.grid(True, which="both")
        plt.yticks(np.arange(0, max(distance[target]), 1))
        plt.xticks(np.arange(0, 91, 10))
        plt.show()

    rank.to_csv('/home/jhoh/DL/DTW_rank_220731_final.csv', index=False)
    distance.to_csv('/home/jhoh/DL/DTW_distance_220731_final.csv', index=False)


main()
