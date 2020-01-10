import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
#from lifelines import KaplanMeierFitter
from sklearn.linear_model import LinearRegression
import seaborn as sns


plt.style.use('ggplot')


# Functions
def kaplan_meier(kmf, df, tumor_type, label):
    na_idx = df.duration[tumor_type].notna()
    T = df.duration[tumor_type][na_idx]
    E = df.death_observed[tumor_type][na_idx]
    kmf.fit(T, event_observed=E, label=label)
    kmf.plot(show_censors=True, ci_show=False, censor_styles={'marker' : '|'})
    print(label, kmf.median_survival_time_, len(T))

def plot_hist(df, feature, xlabel, ylabel):
    plt.hist(df[feature].dropna(), alpha=0.8, edgecolor='white')
    plt.axvline(np.median(df[feature].dropna()), color='k', linestyle='dashed', linewidth=1)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

def plot_box(df, feature, ylabel):
    carc = df[feature][(df.carcinosarcoma == 1)]
    sar = df[feature][(df.carcinosarcoma == 0)]
    plt.boxplot([carc.dropna(), sar.dropna()], notch=False)
    plt.xticks([1,2], ['Karzinosarkom', 'Sarkom'])
    plt.ylabel(ylabel)
    plt.show()

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def plot_violin(df, feature, ylabel):
    carc = df[feature][(df.carcinosarcoma == 1)]
    sar = df[feature][(df.carcinosarcoma == 0)]
    data = [np.array(sorted(carc.dropna())), np.array(sorted(sar.dropna()))]
    parts = plt.violinplot(data, positions=[1,2], showextrema=False)

    for pc in parts['bodies']:
        pc.set_edgecolor('black')
    
    quartile1, medians, quartile3 = zip(np.percentile(data[0], [25, 50, 75]), np.percentile(data[1], [25, 50, 75]))
    whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
    inds = np.arange(1, len(medians) + 1)
    plt.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    plt.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    
    plt.xticks([1,2], ['Karzinosarkom', 'Sarkom'])
    plt.ylabel(ylabel)
    plt.show()

def plot_violin_figo(df, feature, ylabel):
    figo_i = df[feature][(df.figo == 1)]
    figo_ii = df[feature][(df.figo == 2)]
    figo_iii = df[feature][(df.figo == 3)]
    figo_iv = df[feature][(df.figo == 4)]
    data = [np.array(sorted(figo_i.dropna())), np.array(sorted(figo_ii.dropna())), np.array(sorted(figo_iii.dropna())), np.array(sorted(figo_iv.dropna()))]
    parts = plt.violinplot(data, positions=[1, 2, 3, 4], showextrema=False)

    for pc in parts['bodies']:
        pc.set_edgecolor('black')
    
    quartile1, medians, quartile3 = zip(np.percentile(data[0], [25, 50, 75]), np.percentile(data[1], [25, 50, 75]), np.percentile(data[2], [25, 50, 75]), np.percentile(data[3], [25, 50, 75]))
    whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
    inds = np.arange(1, len(medians) + 1)
    plt.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    plt.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    
    plt.xticks([1, 2, 3, 4], ['I', 'II', 'III', 'IV'])
    plt.ylabel(ylabel)
    plt.xlabel('FIGO')
    plt.show()

def count_cat_feature(df, feature, cat, cat_val):
    return [len(df[feature][(df[cat] == cat_val) & (df[feature] == 0)]), len(df[feature][(df[cat] == cat_val) & (df[feature] == 1)])]

def plot_bar_bin(df, feature, ylabel, title):
    carc = count_cat_feature(df, feature, 'carcinosarcoma', 1)
    sar = count_cat_feature(df, feature, 'carcinosarcoma', 0)
    bars1 = [carc[0], sar[0]]
    bars2 = [carc[1], sar[1]]
    plt.bar([0,1], bars1, label='Nein', alpha=0.8)
    plt.bar([0,1], bars2, bottom=bars1, label='Ja', alpha=0.8)
    plt.xticks([0,1], ['Karzinosarkom', 'Sarkom'])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_reg(df, f1, f2, xlabel, ylabel):
    carc_x = df[f1][(df[f1].notna()) & (df[f2].notna()) & (df.carcinosarcoma == 1)]
    carc_y = df[f2][(df[f1].notna()) & (df[f2].notna()) & (df.carcinosarcoma == 1)]
    sar_x = df[f1][(df[f1].notna()) & (df[f2].notna()) & (df.carcinosarcoma == 0)]
    sar_y = df[f2][(df[f1].notna()) & (df[f2].notna()) & (df.carcinosarcoma == 0)]

    sns.regplot(x=carc_x, y=carc_y, label='Karzinosarkom', marker='+')
    sns.regplot(x=sar_x, y=sar_y, label='Sarkom', marker='+')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_scatter(df, f1, f2, xlabel, ylabel):
    carc_x = df[f1][(df[f1].notna()) & (df[f2].notna()) & (df.carcinosarcoma == 1)]
    carc_y = df[f2][(df[f1].notna()) & (df[f2].notna()) & (df.carcinosarcoma == 1)]
    sar_x = df[f1][(df[f1].notna()) & (df[f2].notna()) & (df.carcinosarcoma == 0)]
    sar_y = df[f2][(df[f1].notna()) & (df[f2].notna()) & (df.carcinosarcoma == 0)]

    plt.scatter(x=carc_x, y=carc_y, label='Karzinosarkom', marker='+')
    plt.scatter(x=sar_x, y=sar_y, label='Sarkom', marker='+')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    


# Load data
col_names = ['id_old', 'id_new', 'hospital', 'date_birth', 'age_diagnosis', 'date_diagnosis', 'date_recurrence', 'age_recurrence', 'follow_up_a', 
             'date_death', 'symptoms_first_diagnosis', 'ethnicity', 'bmi', 'hypertension', 'diabetes', 'hypercholesterolemia', 
             'uterus_myomatosus', 'hrt_contraceptive', 'gravida', 'para', 'endometriosis', 'family_history', 'tumor_type', 'figo',
             'T', 'N', 'M', 'M1_loc', 'G', 'R', 'V', 'L', 'Pn', 'size_primarius', 'HR_state', 'ki67', 'mitosis_per_hpf', 'necrosis', 
             'cellular_atypia', 'many_few_cells', 'operation', 'operation_date', 'hysterectomy', 'tube_ovary_removal', 'LNE', 
             'complete_resection', 'further_operations', 'chemo_neo_adjuvant', 'chemo_neo_reason', 'chemo_neo_cytostatics', 
             'chemo_neo_num_cycles', 'chemo_adjuvant', 'chemo_cytostatics', 'chemo_num_cycles', 'radiotherapy', 'anti_hormone_therapy',
             'follow_up_b', 'follow_up_c']
df = pd.read_csv('Daten.csv', sep=',', usecols=list(range(58)), names=col_names, skiprows=1)
pd.set_option('display.max_rows', df.shape[0]+1)


# Clean data: dates
df = df[:-3].drop(['id_old', 'date_birth'], axis=1)
df.hospital[df.index[df.hospital == 'klinikum GT']] = 'Klinikum GT'
df.date_diagnosis = pd.to_datetime(df.date_diagnosis, dayfirst=True)
df.date_recurrence = pd.to_datetime(df.date_recurrence, dayfirst=True)
df.date_death[df.index[df.date_death == 'verstorben']] = np.nan
df.date_death = pd.to_datetime(df.date_death, dayfirst=True)


# Clean data: recurrance
df['duration_recurrance'] = 12 * (df.date_recurrence.dt.year - df.date_diagnosis.dt.year) + (df.date_recurrence.dt.month - df.date_diagnosis.dt.month)
df.duration_recurrance[64] = np.nan # error in recurrance date, difference = - 100 years


# Clean data: follow up
df.follow_up_a[133] = pd.to_datetime('06/07/2015', dayfirst=True)
df['last_follow_up'] = pd.to_datetime(df.follow_up_a, dayfirst=True)
df = df.drop(['follow_up_a', 'follow_up_b', 'follow_up_c'], axis=1)


# Clean data: survival time
df['death_observed'] = df.date_death.notna() * 1
df['last_obs'] = df.date_death
df.last_obs[pd.notna(df.last_follow_up)] = df.last_follow_up[pd.notna(df.last_follow_up)]
df['duration'] = 12 * (df.last_obs.dt.year - df.date_diagnosis.dt.year) + (df.last_obs.dt.month - df.date_diagnosis.dt.month)
df.duration[76] = np.nan  # exclude from analysis because date of death unclear
df.death_observed[76] = 0


# Clean data: binary categorical
d_bin_cat = {'nein' : 0, 'ja' : 1, 'n' : 0, 'j' : 1}
d_hrt_contra = {'nein' : 0, 'ja' : 1, 'keine' : 0, 'HRT' : 1, 'Aromatasehemmer' : 1, 'GnRH-Analoge' : 1, 'Tamoxifen für 3 Jahre' : 1, 'Tamoxifen bis 2010' : 1}
df.hypertension = df.hypertension.map(d_bin_cat)
df.diabetes = df.diabetes.map(d_bin_cat)
df.hypercholesterolemia = df.hypercholesterolemia.map(d_bin_cat)
df.uterus_myomatosus = df.uterus_myomatosus.map(d_bin_cat)
df.hrt_contraceptive = df.hrt_contraceptive.str.strip().map(d_hrt_contra)
df.endometriosis = df.endometriosis.map(d_bin_cat)



# Clean data: multiple categories
d_tumor_type = {'Leiomyosarkom Corpus' : 'Leiomyosarkom',
                'Karzinosarkom Corpus' : 'Karzinosarkom Uterus',
                'Endometriales Stromasarkom' : 'Endometriales Stromasarkom',
                'Karzinosarkom Ovar' : 'Karzinosarkom Ovar',
                'Karzinosarkom Tube' : 'Karzinosarkom Ovar',
                'Karzinosarkom der Zervix' : 'Karzinosarkom Uterus',
                'Tubenkarzinom' : 'Karzinosarkom Ovar',
                'Karzinosarkom Peritoneums' : 'Karzinosarkom Ovar', 
                'Leiomyosarkom Vagina' : 'Leiomyosarkom',
                'Leiomyosarkom der Vagina' : 'Leiomyosarkom', 
                'Leiomyosarkom Ligamentum teres uteri' : 'Leiomyosarkom',
                'Karzinosarkom Ovar rechts' : 'Karzinosarkom Ovar',
                'Adenosarkom' : 'Adenosarkom'}
d_figo = {'IV' : 4, 'IVB' : 4, 'IIIC' : 3, 'II' : 2, 'Ib' : 1, 'Ia' : 1, 'IIIc' : 3, 'III' : 3, 'IIIa' : 3, 'I' : 1, 'IIb' : 2, 'IB' : 1, 
          'IVa' : 4, 'IIa' : 2, 'IIB' : 2, 'Ivb' : 4, 'Ic' : 1, 'IIIb' : 3, 'Iib' : 2, 'IIIB' : 3, 'Iva' : 4, 'IIIA' : 3, 'IA' : 1, 
          'Iia' : 2, 'IVA' : 4, 'IIC' : 2, 'IIc' : 2, 'IVb' : 4, 'IC' : 1}
df.tumor_type = df.tumor_type.str.strip()
df.tumor_type = df.tumor_type.map(d_tumor_type)
kar_uterus = (df.tumor_type == 'Karzinosarkom Uterus')
kar_ovar = (df.tumor_type == 'Karzinosarkom Ovar')
sar_lei = (df.tumor_type == 'Leiomyosarkom')
sar_adeno = (df.tumor_type == 'Adenosarkom')
sar_ess = (df.tumor_type == 'Endometriales Stromasarkom')
df.figo = df.figo.str.strip()
df.figo = df.figo.map(d_figo)


# Set value of patient with carcinosarcoma to 1
df['carcinosarcoma'] = 0
df['carcinosarcoma'][kar_uterus] = 1
df['carcinosarcoma'][kar_ovar] = 1


# Analysis: Kaplan-Meier_Curve
do_kmf = False
if do_kmf:
    kar_uterus = (df.tumor_type == 'Karzinosarkom Uterus')
    kar_ovar = (df.tumor_type == 'Karzinosarkom Ovar')
    sar_lei = (df.tumor_type == 'Leiomyosarkom')
    sar_adeno = (df.tumor_type == 'Adenosarkom')
    sar_ess = (df.tumor_type == 'Endometriales Stromasarkom')
    kmf = KaplanMeierFitter()
    kaplan_meier(kmf, df, kar_uterus, 'Karzinosarkom Uterus')
    kaplan_meier(kmf, df, kar_ovar, 'Karzinosarkom Ovar')
    kaplan_meier(kmf, df, sar_lei, 'Leiomyosarkom')
    kaplan_meier(kmf, df, sar_adeno, 'Adenosarkom')
    kaplan_meier(kmf, df, sar_ess, 'Endometriales Stromasarkom')
    plt.xlabel('Zeit [Monaten]')
    plt.ylabel('Überlebenswahrscheinlichkeit')
    plt.show()


# Analysis: Age at Diagnosis
do_age_diagnosis = False
if do_age_diagnosis:
    plot_hist(df, 'age_diagnosis', 'Alter bei Diagnose', 'Anzahl Patienten')
    plot_violin(df, 'age_diagnosis', 'Alter bei Diagnose')
    print('Median Age', np.median(df.age_diagnosis.dropna()))


# Analysis: BMI
do_bmi = False
if do_bmi:
    plot_hist(df, 'bmi', 'BMI', 'Anzahl Patienten')
    plot_violin(df, 'bmi', 'BMI')
    print('Median BMI', np.median(df.bmi.dropna()))


# Analysis: Uterus Myomatosus
do_um = False
if do_um:
    plot_bar_bin(df, 'uterus_myomatosus', 'Anzahl Patienten', 'Uterus Myomatosus')


# Analysis: Endometriosis
do_em = False
if do_em:
    plot_bar_bin(df, 'endometriosis', 'Anzahl Patienten', 'Endometriosis')

# Analysis: Para
do_para = False
if do_para:
    plt.hist(df.para.dropna(), alpha=0.8, edgecolor='white', bins=np.arange(10))
    plt.xlabel('Parität')
    plt.ylabel('Anzahl Patienten')
    xticklabels = np.array_str(np.arange(9))[1:-1].split(' ')
    plt.xticks(np.arange(9)+0.5, xticklabels)
    plt.show()
    plot_violin(df, 'para', 'Parität')


# Analysis: FIGO and Age
do_age_figo = False
if do_age_figo:
    plot_violin_figo(df, 'age_diagnosis', 'Alter bei Diagnose')


# Analysis: FIGO and BMI
do_bmi_figo = False
if do_bmi_figo:
    plot_violin_figo(df, 'bmi', 'BMI')

# Analysis: FIGO and Uterus Myomatosus
do_um_figo = False
if do_um_figo:
    um_figo_i = count_cat_feature(df, 'uterus_myomatosus', 'figo', 1)
    um_figo_ii = count_cat_feature(df, 'uterus_myomatosus', 'figo', 2)
    um_figo_iii = count_cat_feature(df, 'uterus_myomatosus', 'figo', 3)
    um_figo_iv = count_cat_feature(df, 'uterus_myomatosus', 'figo', 4)
    bars1 = [um_figo_i[0], um_figo_ii[0], um_figo_iii[0], um_figo_iv[0]]
    bars2 = [um_figo_i[1], um_figo_ii[1], um_figo_iii[1], um_figo_iv[1]]
    plt.bar([0,1,2,3], bars1, label='Nein', alpha=0.8)
    plt.bar([0,1,2,3], bars2, bottom=bars1, label='Ja', alpha=0.8)
    plt.xticks([0,1,2,3], ['I', 'II', 'III', 'IV'])
    plt.xlabel('FIGO')
    plt.ylabel('Anzahl Patienten')
    plt.title('Uterus Myomatosus')
    plt.legend()
    plt.show()

# Analysis: Recurrance
do_recc = False
if do_recc:
    plot_hist(df, 'duration_recurrance', 'Dauer bis rezidiv [Monate]', 'Anzahl Patienten')
    plot_violin(df, 'duration_recurrance', 'Dauer bis rezidiv [Monate]')

do_age_recc = True
if do_age_recc:
    plot_reg(df, 'age_diagnosis', 'duration_recurrance', 'Alter bei Diagnose', 'Dauer bis rezidiv [Monate]')

do_bmi_recc = True
if do_bmi_recc:
    plot_reg(df, 'bmi', 'duration_recurrance', 'BMI', 'Dauer bis rezidiv [Monate]')