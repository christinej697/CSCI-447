import pandas as pd

# create frequency tables for a given votes data segment
def votes_frequency(train_set):
    data_struct = [
    [train_set[(train_set['class'] == 'republican') & (train_set['infants'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['water'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['adoption'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['physician'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['salvador'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['religious'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['satellite'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['nicaragua'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['missile'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['immigration'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['synfuels'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['education'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['superfund'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['crime'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['exports'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['south-africa'] == 'y')].shape[0]],
    [train_set[(train_set['class'] == 'republican') & (train_set['infants'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['water'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['adoption'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['physician'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['salvador'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['religious'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['satellite'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['nicaragua'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['missile'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['immigration'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['synfuels'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['education'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['superfund'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['crime'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['exports'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['south-africa'] == 'n')].shape[0]],
    [train_set[(train_set['class'] == 'republican') & (train_set['infants'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['water'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['adoption'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['physician'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['salvador'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['religious'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['satellite'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['nicaragua'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['missile'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['immigration'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['synfuels'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['education'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['superfund'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['crime'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['exports'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['south-africa'] == '?')].shape[0]],
    [train_set[(train_set['class'] == 'democrat') & (train_set['infants'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['water'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['adoption'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['physician'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['salvador'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['religious'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['satellite'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['nicaragua'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['missile'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['immigration'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['synfuels'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['education'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['superfund'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['crime'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['exports'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['south-africa'] == 0)].shape[0]],
    [train_set[(train_set['class'] == 'democrat') & (train_set['infants'] == 1)].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['water'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['adoption'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['physician'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['salvador'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['religious'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['satellite'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['nicaragua'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['missile'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['immigration'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['synfuels'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['education'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['superfund'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['crime'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['exports'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['south-africa'] == 'n')].shape[0]],
    [train_set[(train_set['class'] == 'democrat') & (train_set['infants'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['water'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['adoption'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['physician'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['salvador'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['religious'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['satellite'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['nicaragua'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['missile'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['immigration'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['synfuels'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['education'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['superfund'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['crime'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['exports'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['south-africa'] == '?')].shape[0]]
]
    vote_set_frequency_table = pd.DataFrame(data_struct, columns=["infants","water","adoption","physician","salvador","religious","satellite","nicaragua","missile","immigration","synfuels","education","superfund","crime","exports","south-africa"], index=['y-republican','n-republican','?-republican','y-democrat','n-democrat','?-democrat'])

    return vote_set_frequency_table