import streamlit as st
import pickle
import numpy as np
#import joblib 


def load_model():
    with open('AA_LLC.pkl', "rb") as file:
        data = pickle.load(file)
    return data


data = load_model()
model = data['model']
le_country = data["le_country"]
le_state = data["le_state"]
le_attack = data["le_attack"]
le_target = data["le_target"]
le_group = data["le_group"]
le_weapon = data["le_weapon"]

def show_predict_page():
    st.title("Terrorism Attack Prediction")

    st.write("""### We need some information to predict the attack""")

    countries = (
                    "Somalia",                            
                    "Nigeria",                            
                    "South Africa",                        
                    "Sudan",                                
                    "Democratic Republic of the Congo",   
                    "Kenya",                                
                    "Burundi",                              
                    'Mali',                                 
                    "Angola",                               
                    "Uganda",                               
                    "Mozambique",                                                          
                    "Cameroon",                             
                    "Central African Republic",             
                    "South Sudan",                          
                    "Ethiopia",                            
                    "Rwanda",                              
                    "Namibia",                              
                    "Niger",                               
                    "Senegal",                              
                    "Zimbabwe",                             
                    "Sierra Leone",                        
                    "Chad",                                  
                    "Rhodesia",                              
                    "Ivory Coast",                           
                    "Zambia",                                
                    "Tanzania",                             
                    "Burkina Faso",                         
                    "Zaire", 
    )

    state = (
                    'Banaadir',            
                    'Borno',             
                    'Gauteng',             
                    'Lower Shebelle',       
                    'Unknown',              
                    'KwaZulu-Natal',        
                    'North Kivu',           
                    'Lower Juba',          
                    'Bay',                  
                    'North Darfur',         
                    'Bujumbura Mairie',    
                    'Hiiraan',              
                    'Extreme-North',        
                    'Yobe',                 
                    'Gedo',                 
                    'Benue',                
                    'Northern',             
                    'Adamawa',              
                    'South Darfur'         
                    'Western Cape',        
                    'Plateau',              
                    'Rivers',               
                    'Kaduna',               
                    'Delta',                
                    'Kano',                 
                    'Bari',                
                    'Mudug',                
                    'Middle Shebelle',      
                    'Orientale',            
                    'Eastern Cape'         
                    'Kidal',               
                    'Central',              
                    'Gao',                 
                    'Timbuktu',            
                    'Galguduud',            
                    'Bayelsa',              
                    'North Eastern',       
                    'Bakool',              
                    'Mandera',
        
    )
    attackType = (
                    'Hostage Taking (Kidnapping)', 'Hijacking',
       'Assassination', 'Bombing/Explosion', 'Armed Assault',
       'Facility/Infrastructure Attack',
       'Hostage Taking (Barricade Incident)', 'Unarmed Assault'
        
    )
        
    targetType = (
                    'Military', 'Journalists & Media', 'Government (Diplomatic)',
       'Government (General)', 'Airports & Aircraft', 'Business',
       'Private Citizens & Property', 'Transportation',
       'Violent Political Party', 'Educational Institution', 'Police',
       'Utilities', 'Religious Figures/Institutions', 'Tourists',
       'NGO', 'Telecommunication', 'Food or Water Supply',
       'Maritime', 'Terrorists/Non-State Militia',
    )
    
    groupName = (
                    "South-West Africa People's Organization (SWAPO)",
       'African National Congress (South Africa)',
       'National Union for the Total Independence of Angola (UNITA)',
       'Mozambique National Resistance Movement (MNR)',
       'Inkatha Freedom Party (IFP)',
       'Movement of Democratic Forces of Casamance', 'Hutu extremists',
       'Revolutionary United Front (RUF)', "Lord's Resistance Army (LRA)",
       'Allied Democratic Forces (ADF)', 'Mayi Mayi', 'Fulani extremists',
       'Janjaweed',
       'Movement for the Emancipation of the Niger Delta (MEND)',
       'Democratic Front for the Liberation of Rwanda (FDLR)',
       'Muslim extremists', 'Al-Shabaab',
       'Al-Qaida in the Islamic Maghreb (AQIM)', 'Boko Haram',
       "Sudan People's Liberation Movement - North",
       'Movement for Oneness and Jihad in West Africa (MUJAO)',
       'Ansar al-Dine (Mali)', 'Anti-Balaka Militia',
       "Sudan People's Liberation Movement in Opposition (SPLM-IO)",
       'Niger Delta Avengers (NDA)',
       'Jamaat Nusrat al-Islam wal Muslimin (JNIM)', 
    )
        
        
    weapon = ('Explosives', 'Firearms', 'Incendiary', 'Melee',
       
    )
    suicide = (0,1)
        
        
        
    country = st.selectbox("Country", countries)
    state = st.selectbox("State", state)
    attackType = st.selectbox("AttackType", attackType)
    targetType = st.selectbox("TaretType", targetType)
    groupName = st.selectbox("Group", groupName)
    weapon = st.selectbox("Weapon", weapon)
    sucuide = st.selectbox("Suicide", suicide)

    ok = st.button("Predict Attack")
    if ok:
        X = np.array([[country, state, attackType,targetType,groupName,weapon,suicide]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_state.transform(X[:,1])
        X[:, 2] = le_attack.transform(X[:,2])
        X[:, 3] = le_target.transform(X[:,3])
        X[:, 4] = le_group.transform(X[:,4])
        X[:, 5] = le_weapon.transform(X[:,5])
        #X = X.astype(np.float64)
        #X.reshape(1,-1) 
        #st.write(X)
      

        prediction = model.predict(X)
        if (prediction[0] == 0):
            print('The attack is not successful')
        else:
            print('The attack is successful')
