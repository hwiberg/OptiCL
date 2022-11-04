import itertools

X_cols = ['Pub_Year', 'Asia', 'N_Patient', 'FRAC_MALE', 'AGE_MED',
       'Prior_Palliative_Chemo', 'Primary_Stomach', 'Primary_GEJ', 'ECOG_MEAN']

T_names = ['9-Aminocamptothecin', 'Actinomycin', 'BBR 3438', 
          'BMS-182248-01', 'BOF-A2', 'Bevacizumab', 'Bortezomib', 
          'Bryostatin-1', 'Caelyx', 'Capecitabine', 'Carboplatin', 
          'Carmustine', 'Cetuximab', 'Cisplatin', 'Cyclophosphamide', 
          'Cytarabine', 'DHA-paclitaxel', 'Diaziquone', 'Docetaxel', 
          'Doxifluridine', 'Doxorubicin', 'Epirubicin', 'Erlotinib', 
          'Esorubicin', 'Etoposide', 'Everolimus', 'Flavopiridol', 
          'Fluorouracil', 'Fotemustine', 'Gefitinib', 'Gemcitabine', 
          'Heptaplatin', 'IFN', 'Iproplatin', 'Irinotecan', 
          'Irofulven', 'Ixabepilone', 'Lapatinib', 'Leucovorin', 
          'Levoleucovorin', 'Lovastatin', 'Matuzumab', 'Merbarone', 
          'Methotrexate', 'Mitomycin', 'Mitoxantrone', 'NK105', 
          'OSI-7904L', 'Oxaliplatin', 'PALA', 'PN401', 'Paclitaxel', 
          'Pegamotecan', 'Pemetrexed', 'Pirarubicin', 'Piroxantrone', 
          'Pravastatin', 'Raltitrexed', 'S-1', 'Saracatinib', 
          'Sorafenib', 'Sunitinib', 'Thioguanine', 'Topotecan', 
          'Trastuzumab', 'Triazinate', 'Trimetrexate', 'UFT', 
          'Vincristine', 'Vindesine', 'Vinorelbine', 'methyl-CCNU']

T_cols = [x+y for (y,x) in itertools.product(['_Ind','_Avg','_Inst'], T_names)]

tox_indiv = ['Neutro4', 'Thrombo4', 'Anemia4', 'Lympho4',
       'GINONV_34', 'ALLERGY_34', 'AUDITORY_34', 'CARDIO_34', 'COAGULATION_34',
       'CONSTITUTIONAL_34', 'EPIDERMAL_34', 'ENDOCRINE_34', 'INFECTION_34',
       'METABOLIC_34', 'HEMORRHAGE_34', 'LYMPHATICS_34', 'MUSCLE_34',
       'NEUROLOGICAL_34', 'OCULAR_34', 'PAIN_34', 'PULMONARY_34', 'RENAL_34',
       'VASCULAR_34', 'OTHER_34']

outcomes = ['Neutro4','OTHER_34','GINONV_34','CONSTITUTIONAL_34','INFECTION_34','DLT_PROP','OS']

cols_tox_nb = ['ALLERGY_34', 'AUDITORY_34', 'CARDIO_34', 'COAGULATION_34', '__FATIGUE_34', '__FEVER_34', 
               'CONSTITUTIONAL_34', 'EPIDERMAL_34', 'ENDOCRINE_34', '__ANOREXIA_34', '__CONSTIPATION_34', 
               '__DEHYDRATION_34', '__Diarrhea34', '__ESOPHAGITIS_34', '__STOMATITIS_34', 'GI_34', 'GINONV_34', 
               'HEMORRHAGE_34', '__Infection34', '__FEBRILE_NEUTRO_34', 'INFECTION_34', 'LYMPHATICS_34', 
               '__ELECTROLYTE_34', '__HEPATIC_34', 'METABOLIC_34', 'MUSCLE_34', 'NEUROLOGICAL_34', 
               'OCULAR_34', 'PAIN_34', 'PULMONARY_34', 'RENAL_34', 'VASCULAR_34', 'OTHER_34', 'Prop_34']


outcome_cols = ['Neutro4','Thrombo4','Anemia4', 'Lympho4',
              'BLOOD_4',
              'DLT_PROP',
              'GI_34',
            'GINONV_34', 
            'ALLERGY_34',
            'AUDITORY_34',
            'CARDIO_34', 
            'COAGULATION_34',
            'CONSTITUTIONAL_34',
            'EPIDERMAL_34',
            'ENDOCRINE_34', 
            'INFECTION_34',  
            'METABOLIC_34',
            'HEMORRHAGE_34',
            'LYMPHATICS_34',
            'MUSCLE_34', 
            'NEUROLOGICAL_34', 
            'OCULAR_34', 
            'PAIN_34',
            'PULMONARY_34',
            'RENAL_34',
            'VASCULAR_34',
            'OTHER_34',
            'OS']

