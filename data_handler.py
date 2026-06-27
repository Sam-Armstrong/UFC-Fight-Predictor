"""
Author: Sam Armstrong
Date: 2020-2021

Description: Class that is responsible for handling all the data required for the predictions. These responsibilities include
scraping the data from the web, storing and interacting with this data in CSV format, and creating the training data for the
deep learning model.
"""

import datetime
import pandas
from typing import Union


TRAINING_DATA_CSV = "data/TrainingData.csv"


def load_csv(path: str) -> Union[pandas.DataFrame, None]:
    """
    Load a CSV file, dropping any legacy pandas index column.
    """
    try:
        dataframe = pandas.read_csv(path)
        if "Unnamed: 0" in dataframe.columns:
            dataframe = dataframe.drop(columns=["Unnamed: 0"])
        return dataframe
    except FileNotFoundError:
        return None


def calculateDaysSince(day: str, month: str, year: str) -> int:
    """
    Calculates the days between a given date and the current date
    """
    a = datetime.date(int(year), int(month), int(day))
    b = datetime.date.today()
    days_since = b - a
    days_since = str(days_since)

    if len(days_since.split(" ")) > 1:
        days_since = int(days_since.split(" ")[0])
    else:
        days_since = 0

    return days_since


class Data:
    def __init__(self):
        try:
            self.fight_results = load_csv("data/FightResults.csv")
            self.fight_stats = load_csv("data/FightStats.csv")
            self.fighter_data = load_csv("data/FighterData.csv")
            self.training_data = load_csv(TRAINING_DATA_CSV)
        except:
            raise Exception("One or more of the necessary data files is not present.")

    # Extracted method for finding the average stats of a fighter for the four most recent fights they had prior to a given date
    def findFighterStats(self, name: str, date: str) -> list:
        # Calculates the number of days since the fight took place
        if "/" in date:
            days_since_fight = calculateDaysSince(
                date.split("/")[0], date.split("/")[1], date.split("/")[2]
            )
        else:
            days_since_fight = calculateDaysSince(
                date.split("-")[2], date.split("-")[1], date.split("-")[0]
            )

        # Collects the relevant data and information
        fighter_data = self.fight_stats[self.fight_stats["Name"].str.contains(name)]
        fighter_useful_data = list()
        fighter_raw_info = self.fighter_data[
            self.fighter_data["Name"].str.contains(name)
        ]
        fighter_info = fighter_raw_info.iloc[0]
        height = fighter_info["Height"]
        reach = fighter_info["Reach"]
        age = fighter_info["Age"]

        years_since = days_since_fight // 365

        fighter_useful_data.append(height)
        fighter_useful_data.append(reach)
        fighter_useful_data.append(age - years_since)

        time = 0
        knockdown = 0
        knockdown_taken = 0
        sig_strikes_landed = 0
        sig_strikes_attempted = 0
        sig_strikes_absorbed = 0
        strikes_landed = 0
        strikes_attempted = 0
        strikes_absorbed = 0
        takedowns = 0
        takedown_attempts = 0
        got_takendown = 0
        submission_attempts = 0
        clinch_strikes = 0
        clinch_strikes_taken = 0
        ground_strikes = 0
        ground_strikes_taken = 0
        i = 0

        # Finds the total stats for a fighter so they can be averaged
        for index, row in fighter_data.iterrows():
            date_list = str(row["Date"]).split("/")
            day = date_list[0]
            month = date_list[1]
            year = date_list[2]
            days_since = calculateDaysSince(day, month, year)

            if days_since > days_since_fight and i <= 4:
                i += 1
                time += int(row["Time"])
                knockdown += int(row["Knockdowns"])
                knockdown_taken += int(row["Knockdowns Against"])
                sig_strikes_landed += int(row["Sig Strikes Landed"])
                sig_strikes_attempted += int(row["Sig Strikes Attempted"])
                sig_strikes_absorbed += int(row["Sig Strikes Absorbed"])
                strikes_landed += int(row["Strikes Landed"])
                strikes_attempted += int(row["Strikes Attempted"])
                strikes_absorbed += int(row["Strikes Absorbed"])
                takedowns += int(row["Takedowns"])
                takedown_attempts += int(row["Takedown Attempts"])
                got_takendown += int(row["Got Taken Down"])
                submission_attempts += int(row["Submission Attempts"])
                clinch_strikes += int(row["Clinch Strikes"])
                clinch_strikes_taken += int(row["Clinch Strikes Taken"])
                ground_strikes += int(row["Ground Strikes"])
                ground_strikes_taken += int(row["Ground Strikes Taken"])

        if i <= 4:
            # Doesn't allow the fighter to be compared if they have had fewer than four fights
            raise Exception()

        # Calculates the stats for the fighter, averaged over the total time they have spent in fights (per minute)
        knockdowns_pm = round((knockdown / (time / 60)), 4)
        gets_knockeddown_pm = round((knockdown_taken / (time / 60)), 4)
        sig_strikes_landed_pm = round((sig_strikes_landed / (time / 60)), 4)
        sig_strikes_attempted_pm = round((sig_strikes_attempted / (time / 60)), 4)
        sig_strikes_absorbed_pm = round((sig_strikes_absorbed / (time / 60)), 4)
        strikes_landed_pm = round((strikes_landed / (time / 60)), 4)
        strikes_attempted_pm = round((strikes_attempted / (time / 60)), 4)
        strikes_absorbed_pm = round((strikes_absorbed / (time / 60)), 4)
        strike_accuracy = round((strikes_landed / strikes_attempted), 4)
        takedowns_pm = round((takedowns / (time / 60)), 4)
        takedown_attempts_pm = round((takedown_attempts / (time / 60)), 4)
        gets_takendown_pm = round((got_takendown / (time / 60)), 4)
        submission_attempts_pm = round((submission_attempts / (time / 60)), 4)
        clinch_strikes_pm = round((clinch_strikes / (time / 60)), 4)
        clinch_strikes_taken_pm = round((clinch_strikes_taken / (time / 60)), 4)
        ground_strikes_pm = round((ground_strikes / (time / 60)), 4)
        ground_strikes_taken_pm = round((ground_strikes_taken / (time / 60)), 4)

        # Adds all of the averaged stats to a list
        fighter_useful_data.append(knockdowns_pm)
        fighter_useful_data.append(gets_knockeddown_pm)
        fighter_useful_data.append(sig_strikes_landed_pm)
        fighter_useful_data.append(sig_strikes_attempted_pm)
        fighter_useful_data.append(sig_strikes_absorbed_pm)
        fighter_useful_data.append(strikes_landed_pm)
        fighter_useful_data.append(strikes_attempted_pm)
        fighter_useful_data.append(strikes_absorbed_pm)
        fighter_useful_data.append(strike_accuracy)
        fighter_useful_data.append(takedowns_pm)
        fighter_useful_data.append(takedown_attempts_pm)
        fighter_useful_data.append(gets_takendown_pm)
        fighter_useful_data.append(submission_attempts_pm)
        fighter_useful_data.append(clinch_strikes_pm)
        fighter_useful_data.append(clinch_strikes_taken_pm)
        fighter_useful_data.append(ground_strikes_pm)
        fighter_useful_data.append(ground_strikes_taken_pm)

        return fighter_useful_data

    def createTrainingData(self):
        """
        Creates a set of training data based upon the statistics of each fighter prior to a given fight,
        using the result of the fight as the training label
        """
        if (
            len(self.fight_results) != 0
            and len(self.fight_stats) != 0
            and len(self.fighter_data) != 0
        ):
            training_data = pandas.DataFrame(
                columns=[
                    "Height1",
                    "Reach1",
                    "Age 1",
                    "Knockdowns PM 1",
                    "Gets Knocked Down PM 1",
                    "Sig Strikes Landed PM 1",
                    "Sig Strikes Attempted PM 1",
                    "Sig Strikes Absorbed PM 1",
                    "Strikes Landed PM 1",
                    "Strikes Attempted PM 1",
                    "Strikes Absorbed PM 1",
                    "Strike Accuracy 1",
                    "Takedowns PM 1",
                    "Takedown Attempts PM 1",
                    "Gets Taken Down PM 1",
                    "Submission Attempts PM 1",
                    "Clinch Strikes PM 1",
                    "Clinch Strikes Taken PM 1",
                    "Grounds Strikes PM 1",
                    "Ground Strikes Taken PM 1",
                    "Height 2",
                    "Reach 2",
                    "Age 2",
                    "Knockdowns PM 2",
                    "Gets Knocked Down PM 2",
                    "Sig Strikes Landed PM 2",
                    "Sig Strikes Attempted PM 2",
                    "Sig Strikes Absorbed PM 2",
                    "Strikes Landed PM 2",
                    "Strikes Attempted PM 2",
                    "Strikes Absorbed PM 2",
                    "Strike Accuracy 2",
                    "Takedowns PM 2",
                    "Takedown Attempts PM 2",
                    "Gets Taken Down PM 2",
                    "Submission Attempts PM 2",
                    "Clinch Strikes PM 2",
                    "Clinch Strikes Taken PM 2",
                    "Grounds Strikes PM 2",
                    "Ground Strikes Taken PM 2",
                    "Win",
                    "Loss",
                ]
            )
            all_data = list()

            # Loops through all the fight results and attempts to find the stats for each of the fighters from their
            # four prior fights. This data can then be labelled with the fight result and used to train the neural network
            # model in the 'Predictor' class, when it is called from main.
            for index, row in self.fight_results.iterrows():
                try:
                    date = str(row["Date"])
                    days_since_fight = calculateDaysSince(
                        date.split("/")[0], date.split("/")[1], date.split("/")[2]
                    )
                    name = str(row["Fighter 1"]).strip()
                    name2 = str(row["Fighter 2"]).strip()
                    result = row["Result"]
                    split_dec = row["Split Dec?"]

                    if days_since_fight > 75:  # 0
                        print(date)
                        # Doesn't include any fights that happened before 2010
                        if int(date.split("/")[2]) < 2010:
                            raise Exception()

                        # Finds the stats of the two fighters prior to the date of the given fight occuring
                        fighter1_useful_data = self.findFighterStats(name, date)
                        fighter2_useful_data = self.findFighterStats(name2, date)

                        # Produces a 'one-hot' array describing the outcome of the fight
                        if result == 2 and split_dec == 1:
                            result_array = [0.4, 0.6]
                            opposite_array = [0.6, 0.4]
                        elif result == 1 and split_dec == 1:
                            result_array = [0.6, 0.4]
                            opposite_array = [0.4, 0.6]
                        elif result == 2:
                            result_array = [0, 1]
                            opposite_array = [1, 0]
                        elif result == 1:
                            result_array = [1, 0]
                            opposite_array = [0, 1]
                        else:  # Draw
                            result_array = [0.5, 0.5]
                            opposite_array = [0.5, 0.5]

                        # Concatenates the full training row with the data from both fighters and the 'one-hot' label array
                        full_list1 = (
                            fighter1_useful_data + fighter2_useful_data + result_array
                        )

                        # The training array is also reversed to help the model generalise to trends/reduce model overfitting
                        full_list2 = (
                            fighter2_useful_data + fighter1_useful_data + opposite_array
                        )

                        # Both arrays are added to the training set
                        all_data.append(full_list1)
                        all_data.append(full_list2)

                except:
                    # If there is not the data for four fights accessible for each of the fighters, the training row won't be created
                    pass

            # Adds the training data to the dataframe
            for data in all_data:
                df_len = len(training_data)
                training_data.loc[df_len] = data

            training_data.to_csv(TRAINING_DATA_CSV, index=False)
            self.training_data = load_csv(TRAINING_DATA_CSV)
            print("Finished.")

        else:
            print(
                "One or more of the necessary data files is not present. Please scrape the data using the interface button. "
            )
