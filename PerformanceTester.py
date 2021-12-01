from PyTorchPredictor import Predictor
from Data import Data
import pandas
import numpy as np

def test():
    predictor = Predictor()
    data = Data()

    df = pandas.read_csv('TestOdds.csv')
    i = 0
    returns = 0

    for index, row in df.iterrows():
        try:
            date = row[0]
            fighter1 = row[1]
            fighter2 = row[2]
            result = row[3]
            odds1 = row[4]
            odds2 = row[5]

            prediction_data1 = (data.findFighterStats(fighter1, date)) + (data.findFighterStats(fighter2, date))
            prediction_data2 = (data.findFighterStats(fighter2, date)) + (data.findFighterStats(fighter1, date))

            prediction_data1 = np.array([prediction_data1])
            prediction1 = predictor.predict(prediction_data1)
            prediction_data2 = np.array([prediction_data2])
            prediction2 = predictor.predict(prediction_data2)

            #print(prediction_data1)

            chance1 = round((prediction1[0][0].item() + prediction2[0][1].item()) / 2, 3)
            chance2 = round((prediction1[0][1].item() + prediction2[0][0].item()) / 2, 3)

            est_return1 = round(chance1 * float(odds1), 3)
            est_return2 = round(chance2 * float(odds2), 3)

            if est_return1 > 1 and est_return1 > est_return2: #est_return1 > 1 and 
                chosen_bet = 1
                chosen_odds = odds1
            elif est_return2 > 1 and est_return2 >= est_return1: #est_return2 > 1 and 
                chosen_bet = 2
                chosen_odds = odds2
            else:
                chosen_bet = 0
                chosen_odds = 0

            # if chance1 > chance2:
            #     chosen_bet = 1
            #     chosen_odds = odds1
            # elif chance1 <= chance2:
            #     chosen_bet = 2
            #     chosen_odds = odds2
            # else:
            #     chosen_bet = 0
            #     chosen_odds = 0

            if chosen_bet == 0:
                pass
            elif chosen_bet == int(result):
                i += 1
                returns += chosen_odds
            else:
                i += 1

            # print(fighter1, fighter2)
            # print(chosen_bet, result)
            # print(chance1, chance2)
            # print(est_return1, est_return2)
            # print()

        except Exception as e:
            pass

    print('i: ', i)
    print('Returns: ', returns)
    print('Average Return: ', round(returns / i, 3))




if __name__ == '__main__':
    test()