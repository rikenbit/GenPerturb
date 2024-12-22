import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error

class ModelStatsFC:
    def __init__(self, study, df, pred, pretrained_model):
        self.study = study
        self.df = df
        self.pred = pred

        # Create dataset
        ctrl = self.df.columns[1]
        self.df1 = (self.df.iloc[:, 2:].T - self.df[ctrl]).T.copy()
        self.df1.insert(0, "training", self.df["training"])
        self.df2 = pd.DataFrame(self.pred)
        self.df2.columns = self.df.columns[1:]
        self.df2.index = self.df.index
        self.df2 = (self.df2.T - self.df2[ctrl]).T.drop(ctrl, axis=1).copy()

        # Check FC num
        self.fc_num = self.df.iloc[:, 1:].apply(lambda row: row[(row >= 1) | (row <= -1)].count(), axis=1)

        if pretrained_model == "enformer":
            self.color = cmap = sns.color_palette("deep")[0]
            self.cmap  = "Blues_r"
        elif pretrained_model in ["hyena_dna_tss", "hyena_dna_last", "hyena_dna_mean"]:
            self.color = cmap = sns.color_palette("deep")[1]
            self.cmap  = "Oranges_r"
        elif pretrained_model in ["nucleotide_transformer_tss", "nucleotide_transformer_cls", "nucleotide_transformer_mean"]:
            self.color = cmap = sns.color_palette("deep")[2]
            self.cmap  = "Greens_r"


    def count_cells_above_threshold(self, dataframe, threshold=1):
        count_per_row_plus = dataframe.apply(lambda row: row[(row >= threshold)].count(), axis=1)
        count_per_row_minus = dataframe.apply(lambda row: row[(row <= -threshold)].count(), axis=1)
        return count_per_row_plus, count_per_row_minus

    def save_fold_change_num(self):
        plus2, minus2 = self.count_cells_above_threshold(self.df2.T, threshold=2)
        plus, minus = self.count_cells_above_threshold(self.df2.T, threshold=1)
        plus05, minus05 = self.count_cells_above_threshold(self.df2.T, threshold=0.5)
        fc_num_list = pd.concat([plus2, minus2, plus, minus, plus05, minus05], axis=1)
        fc_num_list = fc_num_list.rename(columns={0: "up_2", 1: "down_2", 2: "up_1", 3: "down_1", 4: "up_05", 5: "down_05"})
        fc_num_list.to_csv(f"figures/{self.study}/prediction_foldchange_num_fc.txt", sep="\t")

    def cal_correlation(self, _df1, _df2):
        r_values = []
        r_pval = []
        slope = []
        mse = []
        for idx in range(0, len(_df1)):
            try:
                a, b, r_value, p_value, std_err = stats.linregress(_df1.iloc[idx, 1:].astype("float32"),
                                                                    _df2.iloc[idx, :])
                r_values.append(r_value)
                r_pval.append(p_value)
                slope.append(a)
            except:
                r_values.append(0)
                r_pval.append(1)
            mse_value = mean_squared_error(_df1.iloc[idx, 1:].astype("float32"), _df2.iloc[idx, :])
            mse.append(mse_value)
        r_summary = pd.concat([_df1.loc[:, "training"].reset_index(), pd.Series(r_values), pd.Series(r_pval),
                               pd.Series(slope), pd.Series(mse), pd.Series(self.fc_num.reset_index()[0])], axis=1)
        r_summary.columns = ["Gene", "training", "Correlation", "pval", "slope", "MSE", "FC_gene_num"]
        return r_summary.drop_duplicates()

    def cal_correlation_pert(self, _df1, _df2):
        r_summary_list = []
        for val in ["train", "val", "test"]:
            df1_subset = _df1[_df1['training'] == val].iloc[:, 1:].T
            df2_subset = _df2[_df2['training'] == val].iloc[:, 1:].T
            r_values, r_pval, slope, mse = [], [], [], []
            for idx in range(len(df1_subset)):
                try:
                    a, b, r_value, p_value, std_err = stats.linregress(df1_subset.iloc[idx, :], df2_subset.iloc[idx, :])
                    mse_value = mean_squared_error(df1_subset.iloc[idx, :], df2_subset.iloc[idx, :])
                    r_values.append(r_value)
                    r_pval.append(p_value)
                    slope.append(a)
                    mse.append(mse_value)
                except:
                    r_values.append(0)
                    r_pval.append(1)
                    slope.append(0)
                    mse.append(0)
            each_summary = pd.DataFrame({
                "Gene": df1_subset.reset_index()["index"],
                "Correlation": r_values,
                "pval": r_pval,
                "slope": slope,
                "MSE": mse,
                "training": [val] * len(df1_subset)
            })
            r_summary_list.append(each_summary)
        r_summary = pd.concat(r_summary_list, axis=0)
        return r_summary.drop_duplicates()


    def plot_boxplot_by_exp(self, cor_acperts, output="_across_perts", yliml=0):
        plt.figure(figsize=(7/2.54, 4/2.54), dpi=300)
        plt.rcParams["font.size"] = 6
        dot_size = 0.3
        ax = sns.boxplot(data=cor_acperts, x="training", y="Correlation", hue='Mean', width=0.6, fliersize=0,
            hue_order=["Very High", "High", "Medium", "Low", "Very Low"], palette=self.cmap, order=["train", "val", "test"])
        ax.set_ylim(min(yliml, 0), 1)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Expression")
        plt.title('Correlation across perturbations (FC)', fontsize=6)
        plt.tight_layout()
        plt.savefig(f'figures/{self.study}/boxplot_fc/Correlation_mean_expression{output}.svg')
        plt.clf()
        plt.close()


    def plot_scatter(self, data, row_name, correlation, value="fold change", prefix=""):
        plt.figure(figsize=(4/2.54, 4/2.54), dpi=300)
        plt.rcParams["font.size"] = 6
        ax = sns.scatterplot(x="obs", y="pred", data=data, s=5, color=self.color)
        ax.annotate(str(f"r = {correlation}"), xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=6, ha='left', va='top', color='black')
        plt.xlabel(f'Real {value}')
        plt.ylabel(f'Pred {value}')
        plt.title(f'{list(set(data["study"]))[0]}\n{row_name}')
        plt.tight_layout()
        plt.savefig(f'figures/{self.study}/scatterplot_fc/{prefix}{row_name}.png')
        plt.clf()
        plt.close()

    def loop_plot_scatter(self, cor, prefix=""):
        rows = cor["Gene"].to_list()
        for row_name in rows:
            print(row_name)
            try:
                data1 = self.df1.loc[row_name,:][1:]
                data2 = self.df2.loc[row_name,:]
                data = pd.concat([data1, data2], axis=1)
                data.columns = ["obs", "pred"]
                data["study"] = [i.split(".")[0] for i in data.index]
                correlation = cor.query('Gene == @row_name')["Correlation"].values[0].round(3)
                self.plot_scatter(data, row_name, correlation, prefix=prefix)
            except:
                continue

    def loop_plot_scatter_pert(self, cor, prefix=""):
        rows = cor["Gene"].to_list()
        test_genes = self.df1.query('training == "test"').index.to_list()
        for col_name in rows:
            print(col_name)
            try:
                data1 = self.df1.loc[test_genes, col_name].T
                data2 = self.df2.loc[test_genes, col_name].T
                data = pd.concat([data1, data2], axis=1)
                data.columns = ["obs", "pred"]
                data["study"] = col_name.split(".")[0]
                correlation = cor.query('Gene == @col_name')["Correlation"].values[0].round(3)
                self.plot_scatter(data, col_name, correlation, prefix=prefix)
            except:
                continue

    def main(self, load_stats=False):
        if load_stats == False:
            os.makedirs(f'figures/{self.study}/cor_matrix_fc', exist_ok=True)
            
            self.save_fold_change_num()

            cor_acperts = self.cal_correlation(self.df1, self.df2)
            cor_acperts = pd.merge(cor_acperts, pd.DataFrame({"var":np.var(self.df.iloc[:,1:], axis=1)}), left_on="Gene", right_index=True)
            cor_acperts = pd.merge(cor_acperts, pd.DataFrame({"mean":np.mean(self.df.iloc[:,1:], axis=1)}), left_on="Gene", right_index=True)
            cor_acperts['Variance'] = pd.qcut(cor_acperts["var"], q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])
            cor_acperts['Mean'] = pd.qcut(cor_acperts["mean"], q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])
            self.cor_acperts = cor_acperts.sort_values("Correlation", ascending=False)
            self.cor_acperts.to_csv(f'figures/{self.study}/cor_matrix_fc/correlation_across_perts.txt', sep="\t", index=False)

            df3 = pd.concat([self.df1.loc[:, "training"], self.df2], axis=1)
            cor_acgenes = self.cal_correlation_pert(self.df1, df3)
            self.cor_acgenes = cor_acgenes.sort_values("Correlation", ascending=False)
            self.cor_acgenes.to_csv(f'figures/{self.study}/cor_matrix_fc/correlation_across_genes.txt', sep="\t", index=False)

        elif load_stats == True:
            self.cor_acperts = pd.read_csv(f'figures/{self.study}/cor_matrix_fc/correlation_across_perts.txt', sep="\t")
            self.cor_acgenes = pd.read_csv(f'figures/{self.study}/cor_matrix_fc/correlation_across_genes.txt', sep="\t")

        os.makedirs(f'figures/{self.study}/boxplot_fc', exist_ok=True)
        self.plot_boxplot_by_exp(self.cor_acperts, yliml=self.cor_acperts["Correlation"].min(), output="_across_perts")

        os.makedirs(f'figures/{self.study}/scatterplot_fc', exist_ok=True)
        self.loop_plot_scatter(self.cor_acperts.query('training == "test"').sort_values("Correlation", ascending=False).head(20).loc[:,["Gene","Correlation"]], prefix="")
        self.loop_plot_scatter(self.cor_acperts.query('training == "test"').sort_values("Correlation", ascending=False).tail(10).loc[:,["Gene","Correlation"]], prefix="00_")
        pickup_genes = ['ENOX2', 'NUMB', 'GIPR', 'PLD4', 'GPR65', 'MRPL52']
        self.loop_plot_scatter(self.cor_acperts.query('training == "test"').query('Gene in @pickup_genes').loc[:,["Gene","Correlation"]], prefix="")


        self.loop_plot_scatter_pert(self.cor_acgenes.query('training == "test"').sort_values("Correlation", ascending=False).head(10).loc[:,["Gene","Correlation"]], prefix="")
        self.loop_plot_scatter_pert(self.cor_acgenes.query('training == "test"').sort_values("Correlation", ascending=False).tail(10).loc[:,["Gene","Correlation"]], prefix="00_")
        ctrl = self.df2.columns[0]
        self.loop_plot_scatter_pert(self.cor_acgenes.query('training == "test" & Gene == @ctrl').loc[:,["Gene","Correlation"]], prefix="")



