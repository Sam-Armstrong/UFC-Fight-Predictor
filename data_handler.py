import datetime
import pandas
from tqdm import tqdm
from typing import Optional, Union

from exceptions import MinFightsException, MissingDataException
from globals import (
    FIGHTER_DATA_CSV,
    MIN_FIGHTS,
    RESULTS_CSV,
    STATS_CSV,
    TRAINING_DATA_CSV,
    VERBOSE,
)


def load_csv(path: str) -> Union[pandas.DataFrame, None]:
    """
    Load a CSV file, dropping any legacy index column
    """
    try:
        dataframe = pandas.read_csv(path)
        if "Unnamed: 0" in dataframe.columns:
            dataframe = dataframe.drop(columns=["Unnamed: 0"])
        return dataframe
    except FileNotFoundError:
        return None


def calculate_days_since(day: str, month: str, year: str) -> int:
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
        self.fight_results = load_csv(RESULTS_CSV)
        self.fight_stats = load_csv(STATS_CSV)
        self.fighter_data = load_csv(FIGHTER_DATA_CSV)
        self.training_data = load_csv(TRAINING_DATA_CSV)

    def find_fighter_stats(self, name: str, date: str, min_fights: int = 3) -> list:
        """
        Find the average stats of a fighter for the four most recent fights they had prior to a given date
        """
        # calculates the number of days since the fight took place
        if "/" in date:
            days_since_fight = calculate_days_since(
                date.split("/")[0], date.split("/")[1], date.split("/")[2]
            )
        else:
            days_since_fight = calculate_days_since(
                date.split("-")[2], date.split("-")[1], date.split("-")[0]
            )

        # collects the relevant data and information
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

        # finds the total stats for a fighter so they can be averaged
        for index, row in fighter_data.iterrows():
            date_list = str(row["Date"]).split("/")
            day = date_list[0]
            month = date_list[1]
            year = date_list[2]
            days_since = calculate_days_since(day, month, year)

            if days_since > days_since_fight and i <= min_fights:
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

        if i <= min_fights:
            # doesn't allow the fighter to be compared if they have had fewer than min_fights fights
            raise MinFightsException(
                f"Fighter {name} had fewer than {min_fights} fights at {date}"
            )

        # calculates the stats for the fighter, averaged over the total time they have spent in fights (per minute)
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

        # adds all of the averaged stats to a list
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

    def create_training_data(self, min_fights: Optional[int] = None):
        """
        Creates a set of training data based upon the statistics of each fighter prior to a given fight,
        using the result of the fight as the training label

        Args:
            min_fights: int or None
                The minimum number of fights a fighter must have had to be considered for training.
                Default to MIN_FIGHTS from globals.py
        """
        if any(
            [
                df is None or len(df) == 0
                for df in [self.fight_results, self.fight_stats, self.fighter_data]
            ]
        ):
            raise MissingDataException()

        if min_fights is None:
            min_fights = MIN_FIGHTS

        training_data = pandas.DataFrame(
            columns=[
                "Height 1",
                "Reach 1",
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
                "Draw",
            ]
        )
        all_data = list()

        # loop through fight results and find the stats for each of the fighters from their n prior fights
        for _, row in tqdm(
            self.fight_results.iterrows(),
            total=len(self.fight_results),
            desc="Creating training data",
            unit="fight",
        ):
            date = str(row["Date"])
            days_since_fight = calculate_days_since(
                date.split("/")[0], date.split("/")[1], date.split("/")[2]
            )
            name1 = str(row["Fighter 1"]).strip()
            name2 = str(row["Fighter 2"]).strip()
            result = row["Result"]
            split_dec = row["Split Dec?"]

            if date > "01/01/2010":

                # finds the stats of the two fighters prior to the date of the given fight occuring
                try:
                    fighter1_useful_data = self.find_fighter_stats(
                        name1, date, min_fights
                    )
                    fighter2_useful_data = self.find_fighter_stats(
                        name2, date, min_fights
                    )
                except Exception as e:
                    if VERBOSE:
                        tqdm.write(f"Skipping fight: {e}")
                    continue

                # one-hot array describing the outcome of the fight (win, loss, draw)
                result_array, opposite_result_array = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
                result_array[result - 1] = 1.0
                opposite_result_array[2 if result == 3 else result % 2] = 1.0

                # concatenates the full training row with the data from both fighters and the 'one-hot' label array
                full_list1 = fighter1_useful_data + fighter2_useful_data + result_array

                # training array is reversed to help the model generalise/reduce overfitting
                full_list2 = (
                    fighter2_useful_data + fighter1_useful_data + opposite_result_array
                )

                all_data.append(full_list1)
                all_data.append(full_list2)

        # add training data to the dataframe
        for data in tqdm(all_data, desc="Building training dataframe", unit="row"):
            df_len = len(training_data)
            training_data.loc[df_len] = data

        training_data.to_csv(TRAINING_DATA_CSV, index=False)
        self.training_data = load_csv(TRAINING_DATA_CSV)


if __name__ == "__main__":
    data = Data()
    data.create_training_data()
    print(f"Training data length: {len(data.training_data)}")
