import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
#from lifelines import KaplanMeierFitter, CoxPHFitter
#from lifelines.datasets import load_rossi
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

def calc_rec_rate_bin(df, cat):
    r0 = sum(df.rec_bool[df[cat] == 0]) / len(df.rec_bool[df[cat] == 0])
    r1 = sum(df.rec_bool[df[cat] == 1]) / len(df.rec_bool[df[cat] == 1])
    return np.array([r0, r1])

def calc_rec_rate(df, cat, vals):
    rec_rates = []
    for i in range(len(vals)-1):
        idx = (df[cat] >= vals[i]) & (df[cat] < vals[i+1])
        if len(df.rec_bool[idx]) > 0:
            rec_rates.append(sum(df.rec_bool[idx]) / len(df.rec_bool[idx]))
        else:
            rec_rates.append(0)
    return np.array(rec_rates)

def plot_bar_rec(df, rec_rates, xticklabels, xlabel):
    x = np.arange(0, len(rec_rates))
    plt.bar(x, rec_rates*100, alpha=0.8)
    plt.xticks(x, xticklabels)
    plt.ylabel('Rezidivrate [%]')
    plt.xlabel(xlabel)
    plt.ylim((0,100))
    plt.show()

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

def plot_violin(df, feature, cat, ylabel, xticklabels):
    cat1 = df[feature][(df[cat] == 1)]
    cat2 = df[feature][(df[cat] == 0)]
    data = [np.array(sorted(cat1.dropna())), np.array(sorted(cat2.dropna()))]
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
    
    plt.xticks([1,2], [xticklabels[0], xticklabels[1]])
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
df['rec_bool'] = df.date_recurrence.notnull().map(lambda x: int(x))


# Clean data: follow up
df.follow_up_a[133] = pd.to_datetime('06/07/2015', dayfirst=True)
df['last_follow_up'] = pd.to_datetime(df.follow_up_a, dayfirst=True)
df = df.drop(['follow_up_a', 'follow_up_b', 'follow_up_c'], axis=1)


# Clean data: survival time
df['death_observed'] = df.date_death.notna() * 1
df['last_obs'] = df.date_death
df.last_obs[pd.notna(df.last_follow_up)] = df.last_follow_up[pd.notna(df.last_follow_up)]
df.loc[:, 'duration'] = 12 * (df.last_obs.dt.year - df.date_diagnosis.dt.year) + (df.last_obs.dt.month - df.date_diagnosis.dt.month)
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
d_T = {'pT1b' : 1, 'pT2b' : 2, 'pT3c' : 3, 'pT3b' : 3, 'pT1a' : 1, 'pT2' : 2, 'pT1c' : 1, 'pT3a' : 3, 'pT2c' : 2, 'pT2a' : 2, 'pT3' : 3,
       'pT4' : 4, 'pT1' : 1, 'pt2' : 2, 'ypT3c' : 3}
d_M = {'M1' : 1, 'M0' : 0, 'pM1' : 1}
d_L = {'L0' : 0, 'L1' : 1}
d_primarius = {'10 cm' : 10.0, '2,5 cm' : 2.5, '6 cm' : 6.0, '4 cm' : 4.0, '10cm' : 10.0, '8 cm' : 8.0, '7 cm' : 7.0, '4.5 cm' : 4.5,
               '13 cm' : 13.0, '3 cm' : 3.0, '100x100x40' : 10.0, '120x100x80' : 12.0, '10x7x7cm' : 10.0, '10x14 cm' : 14.0, 
               '55mm' : 5.5, '6.5 cm' : 6.5, '160mm' : 16.0, '160 x 70 x 50 mm' : 16.0, '115' : 11.5, '110x100x80 mm' : 11.0, 
               '6,5cm' : 6.5, '4,2x1,2cm' : 4.2, 'max Tumorgröße 59mm' : 5.9, '40 x 45 x  53 mm' : 4.0, '195x130x75mm' : 19.5, 
               '12x9cm' : 12.0, '60 x  55 x 55 mm' : 6.0, '230 x 200 x 110 mm' : 23.0, '120 x 70 x 40 mm' : 12.0, '130x100x100 mm' : 13.0,
               '18 cm' : 18.0, '195x175x105mm' : 19.5, '55x65x100 mm' : 5.5, '75x80x60 cm' : 8.0, '15 cm' : 15.0,
               'mind. 90 mm Durchmesser' : 9.0, '90x70x55mm' : 9.0, '4,3x4x4cm' : 4.3, '19 mm Durchmesser' : 1.9, '50 x 40 x 35 mm' : 5.0,
               '85x60x45 mm' : 8.5, '220 x 110 x 80 mm' : 22.0, '47mm' : 4.7, '9cm' : 9.0, 'Einzelne Knoten bis 5 cm' : 5.0, 
               '5,5x7,5x8,5cm' : 8.5, '3,8 cm' : 3.8, '250 mm Durchmesser' : 25.0, '130x120x100mm' : 13.0, '80' : 8.0, '130' : 13.0, 
               '7,5 cm' : 7.5, '190x130x100 mm' : 19.0, '0,4 bis 1,6' : 1.6, '6-7cm' : 7.0, '125x105x30' : 12.5, '4,2 cm' : 4.2, 
               '100x90x90' : 10.0, '60x60x50' : 6.0, '16 cm' : 16.0, '220x80x70 mm' : 22.0, '70x50x30mm' : 7.0, '56' : 5.6,
               '200x100x40mm' : 20.0, '160x130x90mm' : 16.0, '4x5cm' : 4.0, '120x70x20mm' : 12.0, '8x4x3' : 8.0, '5.5 cm' : 5.5,
               '40x27x30 mm' : 4.0, '100 x 90 x 70 mm' : 10.0, '45x40x27' : 4.5, '65x60x50 mm' : 6.5, '14x8x6,5 cm' : 14.0,
               '45x48x35mm' : 4.5, '17,5x13,5x13' : 17.5, '410x200x90mm' : 41.0, '>5cm (Tumor morcelliert)' : 5.0, '70x60x70' : 7.0,
               '100x90x60 mm' : 10.0, '4x3 cm' : 4.0, '320 x 195 x 130 mm' : 32.0, '33mm Durchmesser' : 3.3, '210x140x50mm' : 21.0,
               '12 cm' : 12.0, '11 cm' : 11.0, '64 x 46 x 30mm' : 6.4, '3x4 cm' : 3.0, '35 x 37 x ca. 38 mm.' : 3.8, '48x38mm' : 4.8,
               '35mm' : 3.5, '6x10x12cm' : 12.0, '6cm' : 6.0, '5x3,5cm' : 5.0, '2 cm' : 2.0, '10 x 14 x 11 cm' : 14.0, 
               '290x270bis30' : 29.0, '35x65x45 mm' : 6.5, '40x35x25mm' : 4.0, '170x120x120 mm' : 17.0, '3,5 cm' : 3.5, '9 cm' : 9.0}
df.tumor_type = df.tumor_type.str.strip()
df.tumor_type = df.tumor_type.map(d_tumor_type)
kar_uterus = (df.tumor_type == 'Karzinosarkom Uterus')
kar_ovar = (df.tumor_type == 'Karzinosarkom Ovar')
sar_lei = (df.tumor_type == 'Leiomyosarkom')
sar_adeno = (df.tumor_type == 'Adenosarkom')
sar_ess = (df.tumor_type == 'Endometriales Stromasarkom')
df.figo = df.figo.str.strip()
df.figo = df.figo.map(d_figo)
df['T'] = df['T'].str.strip()
df['T'] = df['T'].map(d_T)
df['N'] = df['N'].str.split(' ')
df['N'] = df['N'].dropna().map(lambda x: float(x[0][-1]))
df['M'] = df['M'].map(d_M)
df['G'] = df['G'].dropna().map(lambda x: float(x[1]))
df['R'] = df['R'].dropna().map(lambda x: float(x[1]))
df['V'] = df['V'].dropna().map(lambda x: float(x[1]))
df['L'] = df['L'].map(d_L)
df['Pn'] = df['Pn'].dropna().map(lambda x: float(x[2]))
df['size_primarius'] = df['size_primarius'].str.strip()
df['size_primarius'] = df['size_primarius'].map(d_primarius)


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
    plot_violin(df, 'age_diagnosis', 'carcinosarcoma', 'Alter bei Diagnose', ['Karzinosarkom', 'Sarkom'])
    print('Median Age', np.median(df.age_diagnosis.dropna()))


# Analysis: BMI
do_bmi = False
if do_bmi:
    plot_hist(df, 'bmi', 'BMI', 'Anzahl Patienten')
    plot_violin(df, 'bmi', 'carcinosarcoma', 'BMI', ['Karzinosarkom', 'Sarkom'])
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
    plot_violin(df, 'para', 'carcinosarcoma', 'Parität', ['Karzinosarkom', 'Sarkom'])

# Analysis: Size Primarius
do_primarius = False
if do_primarius:
    plot_hist(df, 'size_primarius', 'Größe Primarius [cm]', 'Anzahl Patienten')
    plot_violin(df, 'size_primarius', 'carcinosarcoma', 'Größe Primarius [cm]', ['Karzinosarkom', 'Sarkom'])
    print('Median Primarius Size', np.median(df.size_primarius.dropna()))


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
    plot_violin(df, 'duration_recurance', 'carcinosarcoma', 'Dauer bis rezidiv [Monate]', ['Karzinosarkom', 'Sarkom'])

do_age_recc = False
if do_age_recc:
    plot_reg(df, 'age_diagnosis', 'duration_recurrance', 'Alter bei Diagnose', 'Dauer bis rezidiv [Monate]')

do_bmi_recc = False
if do_bmi_recc:
    plot_reg(df, 'bmi', 'duration_recurrance', 'BMI', 'Dauer bis rezidiv [Monate]')

do_hypertension_recc = False
if do_hypertension_recc:
    plot_violin(df, 'duration_recurrance', 'hypertension', 'Dauer bis rezidiv [Monate]', ['Hypertonie ja', 'Hypertonie nein'])

do_diabetes_recc = False
if do_diabetes_recc:
    plot_violin(df, 'duration_recurrance', 'diabetes', 'Dauer bis rezidiv [Monate]', ['Diabetes ja', 'Diabetes nein'])

do_hypercholesterolemia_recc = False
if do_hypercholesterolemia_recc:
    plot_violin(df, 'duration_recurrance', 'hypercholesterolemia', 'Dauer bis rezidiv [Monate]', ['Hypercholestrinämie ja', 'Hypercholestrinämie nein'])

do_figo_recc = False
if do_figo_recc:
    plot_violin_figo(df, 'duration_recurrance', 'Dauer bis rezidiv [Monate]')

do_primarius_recc = False
if do_primarius_recc:
    plot_reg(df, 'size_primarius', 'duration_recurrance', 'Größe Primarius [cm]', 'Dauer bis rezidiv [Monate]')

do_rec_rate = False
if do_rec_rate:
    rec_rates = calc_rec_rate_bin(df, 'carcinosarcoma')
    plot_bar_rec(df, rec_rates, ['Sarkom', 'Karzinosarkom'], '')

do_rec_rate_hypertension = False
if do_rec_rate_hypertension:
    rec_rates = calc_rec_rate_bin(df, 'hypertension')
    plot_bar_rec(df, rec_rates, ['Nein', 'Ja'], 'Hypertonie')

do_rec_rate_diabetes = False
if do_rec_rate_diabetes:
    rec_rates = calc_rec_rate_bin(df, 'diabetes')
    plot_bar_rec(df, rec_rates, ['Nein', 'Ja'], 'Diabetes')

do_rec_rate_hypercholesterolemia = False
if do_rec_rate_hypercholesterolemia:
    rec_rates = calc_rec_rate_bin(df, 'hypercholesterolemia')
    plot_bar_rec(df, rec_rates, ['Nein', 'Ja'], 'Hypercholesterinämie')

do_rec_rate_age = False
if do_rec_rate_age:
    step = 5
    vals = np.arange(0,100,step)
    labels = [str(v) for v in vals[:-1] + step/2]
    rec_rates = calc_rec_rate(df, 'age_diagnosis', vals)
    plot_bar_rec(df, rec_rates, labels, 'Alter bei Diagnose')

do_rec_rate_bmi = False
if do_rec_rate_bmi:
    step = 5
    vals = np.arange(0,50,step)
    labels = [str(v) for v in vals[:-1] + step/2]
    rec_rates = calc_rec_rate(df, 'bmi', vals)
    plot_bar_rec(df, rec_rates, labels, 'BMI')

do_rec_rate_primarius = False
if do_rec_rate_primarius:
    vals = np.arange(0,50,2.5)
    labels = [str(v) for v in vals[:-1] + 1.25]
    rec_rates = calc_rec_rate(df, 'bmi', vals)
    plot_bar_rec(df, rec_rates, labels, 'Größe Primarius [cm]')

do_rec_rate_figo = False
if do_rec_rate_figo:
    r1 = sum(df.rec_bool[df['figo'] == 1]) / len(df.rec_bool[df['figo'] == 1])
    r2 = sum(df.rec_bool[df['figo'] == 2]) / len(df.rec_bool[df['figo'] == 2])
    r3 = sum(df.rec_bool[df['figo'] == 3]) / len(df.rec_bool[df['figo'] == 3])
    r4 = sum(df.rec_bool[df['figo'] == 4]) / len(df.rec_bool[df['figo'] == 4])
    plot_bar_rec(df, np.array([r1, r2, r3, r4]), ['I', 'II', 'III', 'IV'], 'FIGO')

corr = df[['age_diagnosis', 'rec_bool', 'bmi', 'hypertension', 'diabetes', 'hypercholesterolemia', 
           'uterus_myomatosus', 'carcinosarcoma', 'hrt_contraceptive', 'gravida', 'para', 'endometriosis', 'figo', 'T', 'N', 'M', 'G', 'R', 'V', 
           'L', 'size_primarius']].corr()
ticklabels = ['Alter', 'Rezidiv', 'BMI', 'Hypertonie', 'Diabetes', 'Hypercholesterinämie', 
              'Uterus Myomatosus', 'Tumortyp', 'Kontrazeption', 'Gravida', 'Parität', 'Endometriose', 'FIGO', 'T', 'N', 'M', 'G', 'R', 'V', 
              'L', 'Größe Primarius']
sns.heatmap(corr, xticklabels=ticklabels, yticklabels=ticklabels, cmap='coolwarm', vmin=-1, vmax=1)
plt.tight_layout()
plt.show()


#print(df.age_diagnosis.median())
"""
cph = CoxPHFitter()
dur_not_na_idx = df.duration.notna()
df_cph = df[dur_not_na_idx]
df_cph.age_diagnosis = df_cph.age_diagnosis.fillna(df_cph.age_diagnosis.median())
df_cph.bmi = df_cph.bmi.fillna(df_cph.bmi.median())
#print(pd.isnull(df_cph[['duration', 'death_observed', 'age_diagnosis', 'bmi']]))
cph.fit(df_cph[['duration', 'death_observed', 'age_diagnosis', 'bmi']], duration_col='duration', event_col='death_observed')
print(cph.summary)
"""

