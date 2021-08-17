"""
Author: Sam Armstrong
Date: 2020-2021

Description: Class that is responsible for handling all the data required for the predictions. These responsibilities include 
scraping the data from the web, storing and interacting with this data in CSV format, and creating the training data for the 
deep learning model.
"""

import bs4, requests, datetime, pandas

# Calculates the days between a given date and the current date
def calculateDaysSince(day, month, year):
    a = datetime.date(int(year), int(month), int(day))
    b = datetime.date.today()
    days_since = b - a
    days_since = str(days_since)

    if len(days_since.split(' ')) > 1:
        days_since = int(days_since.split(' ')[0])
    else:
        days_since = 0

    return days_since


class Data:
    def __init__(self):
        # Attempts to find all the data files required stored in CSV files
        # Any files that can't be found are set to None objects

        try:
            self.fight_results = pandas.read_csv('FightResults.csv')
        except:
            self.fight_results = None

        try:
            self.fight_stats = pandas.read_csv('FightStats.csv')
        except:
            self.fight_stats = None

        try:
            self.fighter_data = pandas.read_csv('FighterData.csv')
        except:
            self.fighter_data = None

        try:
            self.training_data = pandas.read_csv('TrainingData.csv')
        except:
            self.training_data = None


    # Extracted method for finding the average stats of a fighter for the four most recent fights they had prior to a given date
    def findFighterStats(self, name1, date):
        # Calculates the number of days since the fight took place
        try:
            days_since_fight = calculateDaysSince(date.split('/')[0], date.split('/')[1], date.split('/')[2])
        except:
            days_since_fight = calculateDaysSince(date.split('-')[2], date.split('-')[1], date.split('-')[0])

        # Collects the relevant data and information
        fighter_data = self.fight_stats[self.fight_stats['Name'].str.contains(name1)]
        fighter_useful_data = list()
        fighter_raw_info = self.fighter_data[self.fighter_data['Name'].str.contains(name1)]
        fighter_raw_info = fighter_raw_info.values.tolist()
        fighter_info = fighter_raw_info[0]
        height = fighter_info[5]
        reach = fighter_info[7]
        age = fighter_info[9]

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
            date_list = row[2].split('/')
            day = date_list[0]
            month = date_list[1]
            year = date_list[2]
            days_since = calculateDaysSince(day, month, year)

            if days_since > days_since_fight and i <= 4:
                i += 1
                time += int(row[3])
                knockdown += int(row[4])
                knockdown_taken += int(row[5])
                sig_strikes_landed += int(row[6])
                sig_strikes_attempted += int(row[7])
                sig_strikes_absorbed += int(row[8])
                strikes_landed += int(row[9])
                strikes_attempted += int(row[10])
                strikes_absorbed += int(row[11])
                takedowns += int(row[12])
                takedown_attempts += int(row[13])
                got_takendown += int(row[14])
                submission_attempts += int(row[15])
                clinch_strikes += int(row[16])
                clinch_strikes_taken += int(row[17])
                ground_strikes += int(row[18])
                ground_strikes_taken += int(row[19])

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



    # Scrapes all the data for past fights from the internet and stores this in separate 'results' and 'stats' CSV files
    def getData(self, window):
        print('Scraping Data... (This could take up to a few hours)')
        inital_url = 'http://www.ufcstats.com/statistics/events/completed?page=all'
        page = requests.get(inital_url)
        soup = bs4.BeautifulSoup(page.content, 'lxml')
        event_urls = list()
        url_list = list()

        print('Stage 1/4')
        window.updateProgress(0) # Updates the progress bar on the GUI

        # Finds the url links to all individual UFC events (these contain the data for the fights at each event)
        for a in soup.findAll('a', href=True, attrs={'class': 'b-link b-link_style_black'}):
            new_url = a['href']
            if new_url not in event_urls:
                event_urls.append(new_url)

        # Finds the links for each individual fight from every event
        for url in event_urls:
            window.updateProgress(round(((event_urls.index(url) + 1) * 100) / len(event_urls), 2))
            page = requests.get(url)
            soup = bs4.BeautifulSoup(page.content, 'lxml')

            for link in soup.findAll('tr', attrs={
                'class': 'b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click'}):
                new_link = link['data-link']
                url_list.append(new_link)

        results_dataframe = pandas.DataFrame(columns=['Date', 'Fighter 1', 'Fighter 2', 'Result', 'Split Dec?'])
        stats_dataframe = pandas.DataFrame(
            columns=['Name', 'Date', 'Time', 'Knockdowns', 'Knockdowns Against', 'Sig Strikes Landed',
                     'Sig Strikes Attempted', 'Sig Strikes Absorbed', 'Strikes Landed', 'Strikes Attempted',
                     'Strikes Absorbed', 'Takedowns', 'Takedown Attempts', 'Got Taken Down', 'Submission Attempts',
                     'Clinch Strikes', 'Clinch Strikes Taken', 'Ground Strikes', 'Ground Strikes Taken'])

        all_info = list()
        all_stats = list()
        month_labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                       'October',
                       'November', 'December']

        print('Stage 2/4')
        window.updateProgress(0)

        # Loops through all the fight urls to collect the relevant data
        for url in url_list:
            window.updateProgress(round(((url_list.index(url) + 1) * 100) / len(url_list), 2))

            try:
                value_index = url_list.index(url)
            except:
                value_index = -1

            try:
                page = requests.get(url)
                soup = bs4.BeautifulSoup(page.content, 'lxml')

                stats = list()
                fighter1_stats = list()
                fighter2_stats = list()
                info_data = list()
                split_dec = 0

                text1 = soup.findAll('i', attrs = {'class': 'b-fight-details__text-item_first'})
                text2 = soup.findAll('i', attrs = {
                    'class': 'b-fight-details__person-status b-fight-details__person-status_style_gray'})

                # Notes if a fight result is a split decision (split decisions may reflect less useful data)
                if 'Decision - S' in text1[0].text or len(text2) > 1:
                    split_dec = 1

                # Finds the stats from the page
                raw_stats = soup.findAll('p', attrs = {'class': 'b-fight-details__table-text'})
                for element in raw_stats:
                    stat = element.text
                    stat = stat.replace('\n', '')
                    stat = stat.replace('  ', '')
                    stats.append(stat)

                name1_list = stats[0].split(' ')
                name2_list = stats[1].split(' ')
                name1 = ''
                name2 = ''

                # Processes the fighter names for consistency
                for word in name1_list:
                    if name1 == '':
                        name1 = name1 + word
                    else:
                        name1 = name1 + ' ' + word

                for word in name2_list:
                    if name2 == '':
                        name2 = name2 + word
                    else:
                        name2 = name2 + ' ' + word

                new_fight_info = list()
                # Collects details about the fight - time and number of rounds
                fight_info = soup.findAll('i', attrs = {'class': 'b-fight-details__text-item'})
                for info in fight_info:
                    info = info.text
                    info = info.replace('\n', '')
                    info = info.replace(' ', '')
                    new_fight_info.append(info)

                rounds = new_fight_info[0].split(':')[1]

                time_list = new_fight_info[1].split(':')
                time_min = time_list[1]
                time_sec = time_list[2]

                # Calculates the total time of the fight (in seconds)
                time = ((int(rounds) - 1) * 300) + (int(time_min) * 60) + int(time_sec)

                # Gathers and processes the fight statistics
                knockdowns1 = stats[2]
                knockdowns2 = stats[3]

                sig_strike1_list = stats[4].split(' of ')
                sig_strikes_landed1 = sig_strike1_list[0]
                sig_strikes_attempted1 = sig_strike1_list[1]
                sig_strike2_list = stats[5].split(' of ')
                sig_strikes_landed2 = sig_strike2_list[0]
                sig_strikes_attempted2 = sig_strike2_list[1]

                strike1_list = stats[8].split(' of ')
                strikes_landed1 = strike1_list[0]
                strikes_attempted1 = strike1_list[1]
                strike2_list = stats[9].split(' of ')
                strikes_landed2 = strike2_list[0]
                strikes_attempted2 = strike2_list[1]

                takedowns1 = stats[10].split(' of ')[0]
                takedown_attempts1 = stats[10].split(' of ')[1]
                takedowns2 = stats[11].split(' of ')[0]
                takedown_attempts2 = stats[11].split(' of ')[1]

                submission_attempts1 = stats[14]
                submission_attempts2 = stats[15]

                clinch_strikes1 = stats[34 + (int(rounds) * 20)].split(' of ')[0]
                clinch_strikes2 = stats[35 + (int(rounds) * 20)].split(' of ')[0]

                ground_strikes1 = stats[36 + (int(rounds) * 20)].split(' of ')[0]
                ground_strikes2 = stats[37 + (int(rounds) * 20)].split(' of ')[0]

                # Finds the result of the fight
                win_loss = soup.findAll('div', attrs = {'class': 'b-fight-details__person'})
                for w in win_loss:
                    l = w.text.replace('\n', '')
                    l = l.split(' ')

                    for x in l:
                        if x == 'W':
                            r = 2
                        if x == 'L':
                            r = 1

                # Finds the date that the fight took place
                date_element = soup.findAll('a', href = True, attrs = {'class': 'b-link'})[0]
                date_url = date_element['href']
                page = requests.get(date_url)
                soup = bs4.BeautifulSoup(page.content, 'lxml')
                raw_date = soup.findAll('li', attrs = {'class': 'b-list__box-list-item'})[0].text
                raw_date = raw_date.replace('\n', '')
                raw_date = raw_date.replace('Date:', '')
                raw_date = raw_date.replace('  ', '')
                raw_date = raw_date.replace(',', '')
                date_list = raw_date.split(' ')
                month = month_labels.index(date_list[0]) + 1
                day = date_list[1]
                year = date_list[2]

                date = str(day) + '/' + str(month) + '/' + str(year)

                # Adds the data to the appropriate lists
                info_data.append(date)
                info_data.append(name1)
                info_data.append(name2)
                info_data.append(r)
                info_data.append(split_dec)

                fighter1_stats.append(name1)
                fighter1_stats.append(date)
                fighter1_stats.append(time)
                fighter1_stats.append(knockdowns1)
                fighter1_stats.append(knockdowns2)
                fighter1_stats.append(sig_strikes_landed1)
                fighter1_stats.append(sig_strikes_attempted1)
                fighter1_stats.append(sig_strikes_landed2)
                fighter1_stats.append(strikes_landed1)
                fighter1_stats.append(strikes_attempted1)
                fighter1_stats.append(strikes_landed2)
                fighter1_stats.append(takedowns1)
                fighter1_stats.append(takedown_attempts1)
                fighter1_stats.append(takedowns2)
                fighter1_stats.append(submission_attempts1)
                fighter1_stats.append(clinch_strikes1)
                fighter1_stats.append(clinch_strikes2)
                fighter1_stats.append(ground_strikes1)
                fighter1_stats.append(ground_strikes2)

                fighter2_stats.append(name2)
                fighter2_stats.append(date)
                fighter2_stats.append(time)
                fighter2_stats.append(knockdowns2)
                fighter2_stats.append(knockdowns1)
                fighter2_stats.append(sig_strikes_landed2)
                fighter2_stats.append(sig_strikes_attempted2)
                fighter2_stats.append(sig_strikes_landed1)
                fighter2_stats.append(strikes_landed2)
                fighter2_stats.append(strikes_attempted2)
                fighter2_stats.append(strikes_landed1)
                fighter2_stats.append(takedowns2)
                fighter2_stats.append(takedown_attempts2)
                fighter2_stats.append(takedowns1)
                fighter2_stats.append(submission_attempts2)
                fighter2_stats.append(clinch_strikes2)
                fighter2_stats.append(clinch_strikes1)
                fighter2_stats.append(ground_strikes2)
                fighter2_stats.append(ground_strikes1)

                # Adds the data for this fight to the accumulated dataset
                all_info.append(info_data)
                all_stats.append(fighter1_stats)
                all_stats.append(fighter2_stats)

            except:
                print('Passing')
                pass

        # Adds the data to the relevant pandas dataframes
        for data in all_info:
            df_len = len(results_dataframe)
            results_dataframe.loc[df_len] = data

        for stat in all_stats:
            df_len = len(stats_dataframe)
            stats_dataframe.loc[df_len] = stat


        # Saves the dataframes to CSV files
        results_dataframe.to_csv('FightResults.csv')
        stats_dataframe.to_csv('FightStats.csv')
        self.fight_results = pandas.read_csv('FightResults.csv')
        self.fight_stats = pandas.read_csv('FightStats.csv')

        return self.fight_results, self.fight_stats



    # Scrapes the data for each individual fighter from the internet and stores this in a CSV file
    def getFighterData(self, window):

        def calculateAge(month, day, year):
            today = datetime.date.today()
            return today.year - year - ((today.month, today.day) < (month, day))

        not_allowed = ['%', 'lbs.', ',', '"', 'Record:', 'Reach:', 'SLpM:', 'Str. Acc.:', 'SApM:', 'Str. Def:',
                      'TD Avg.:',
                      'TD Acc.:', 'TD Def.:', 'Sub. Avg.:', ' \n', '\n', '  ']
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        allChars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                    't', 'u', 'v', 'w', 'x', 'y', 'z']
        url_list = list()
        
        print('Stage 3/4')
        window.updateProgress(round(0))

        # Finds the urls for the webpages of all individual fighterss
        for c in allChars:
            window.updateProgress(round(((allChars.index(c) + 1) * 100) / len(allChars), 2))
            url = ("http://www.ufcstats.com/statistics/fighters?char=%s&page=all" % c)
            page = requests.get(url)
            soup = bs4.BeautifulSoup(page.content, 'lxml')
            for a in soup.findAll('a', href = True, attrs = {'class': 'b-link b-link_style_black'}):
                new_url = a['href']
                if new_url not in url_list:
                    url_list.append(new_url)

        # Initializes the pandas dataframe for storing individual fighter information
        fighters_dataframe = pandas.DataFrame(
            columns = ['Name', 'Wins', 'Losses', 'Draws', 'Height', 'Weight', 'Reach', 'Stance', 'Age', 'SLpM', 'StrAcc',
                     'SApM', 'StrDef', 'TDAvg', 'TDAcc', 'TDDef', 'SubAvg'])

        print('Stage 4/4')
        window.updateProgress(0)

        # Loops through the webpages for each fighter
        for url in url_list:
            window.updateProgress(round(((url_list.index(url) + 1) * 100) / len(url_list), 2))
            try:
                value_index = url_list.index(url)
            except:
                value_index = -1

            print(round(((value_index / len(url_list)) * 100), 1), '% complete')

            # Gets the page content from the url
            page = requests.get(url)
            soup = bs4.BeautifulSoup(page.content, 'lxml')

            # Finds all the useful data for the webpage and stores it in the dataframe
            data = list()
            # Loops through all the h2 elements found on the webpage
            for h2 in soup.findAll('h2'):
                raw_fighter_name = h2.find('span', attrs = {'class': 'b-content__title-highlight'}).text

                for i in range(0, len(not_allowed)):
                    if not_allowed[i] in raw_fighter_name:
                        raw_fighter_name = raw_fighter_name.replace(not_allowed[i], '')
                name_array = raw_fighter_name.split(' ')

                name = name_array[0]
                for y in range(1, len(name_array)):
                    name = name + ' ' + name_array[y]

                record = h2.find('span', attrs = {'class': 'b-content__title-record'}).text
                for i in [' ', '\n', 'Record:']:
                    if i in record:
                        record = record.replace(i, '')

                record = record.split('-')

                wins = record[0]
                losses = record[1]
                draws = record[2]

                # Ensures that 'draws' is in the correct format
                if '(' in draws:
                    draws = draws[0]

                data.append(name)
                data.append(wins)
                data.append(losses)
                data.append(draws)

            # Loops through all the unordered list elements in the webpage
            for ul in soup.findAll('ul'):
                # Loops through all the list item elements in the given webpage list
                for li in ul.findAll('li', attrs = {'class': 'b-list__box-list-item b-list__box-list-item_type_block'}):
                    collected_data = li.text # The text from the given list item

                    for i in range(0, len(not_allowed)):
                        if not_allowed[i] in collected_data:
                            collected_data = collected_data.replace(not_allowed[i], '')

                    # Processes the data accordingly if it represents the height of the fighter
                    if ('Height:' in str(collected_data)) and ('--' not in str(collected_data)):
                        collected_data = collected_data.replace('Height:', '')
                        measurement = collected_data.split("'")
                        
                        # Converts height to centimetres
                        cm1 = int(measurement[0]) * 30.48
                        cm2 = int(measurement[1]) * 2.54
                        collected_data = round((cm1 + cm2), 1)

                    # Processes the data accordingly if it represents the date of birth of the fighter
                    if ('DOB:' in str(collected_data)) and ('--' not in str(collected_data)):
                        collected_data = collected_data.replace('DOB:', '')
                        dateList = collected_data.split(' ')
                        monthStr = str(dateList[0])
                        day = int(dateList[1])
                        year = int(dateList[2])
                        month = 1
                        for x in range(0, len(month_labels)):
                            if month_labels[x] == monthStr:
                                month = x + 1
                        collected_data = int(calculateAge(month, day, year))

                    # Processes the data accordingly if it represents the weight of the fighter
                    if 'Weight:' in str(collected_data):
                        collected_data = collected_data.replace('Weight:', '')
                        collected_data = collected_data.replace('lbs', '')
                        collected_data = collected_data.replace(' ', '')

                    # Processes the data accordingly if it represents the stance of the fighter
                    elif 'STAN' in str(collected_data):
                        if 'Orthodox' in collected_data:
                            collected_data = 1
                        elif 'Southpaw' in collected_data:
                            collected_data = 2
                        else:
                            collected_data = 3

                    collected_data = str(collected_data)

                    # Adds the current piece of data to the 'data' list in string format
                    if (collected_data != '') and ('--' not in collected_data):
                        data.append(collected_data)

            # Adds the fighter data to the dataframe if the data found reflect the full set of data required
            if len(data) == 17:
                df_len = len(fighters_dataframe)
                fighters_dataframe.loc[df_len] = data

        # Saves the dataframe to a CSV file
        fighters_dataframe.to_csv('FighterData.csv')
        self.fighter_data = fighters_dataframe
        print('Finished.')


    # Creates a set of training data based upon the statistics of each fighter prior to a given fight,
    # using the result of the fight as the training label
    def createTrainingData(self, window):
        window.updateProgress(0)

        if len(self.fight_results) != 0 and len(self.fight_stats) != 0 and len(self.fighter_data) != 0:
            training_data = pandas.DataFrame(
                columns = ['Height1', 'Reach1', 'Age 1', 'Knockdowns PM 1', 'Gets Knocked Down PM 1',
                         'Sig Strikes Landed PM 1', 'Sig Strikes Attempted PM 1', 'Sig Strikes Absorbed PM 1',
                         'Strikes Landed PM 1', 'Strikes Attempted PM 1', 'Strikes Absorbed PM 1', 'Strike Accuracy 1',
                         'Takedowns PM 1', 'Takedown Attempts PM 1', 'Gets Taken Down PM 1', 'Submission Attempts PM 1',
                         'Clinch Strikes PM 1', 'Clinch Strikes Taken PM 1', 'Grounds Strikes PM 1',
                         'Ground Strikes Taken PM 1', 'Height 2', 'Reach 2', 'Age 2', 'Knockdowns PM 2',
                         'Gets Knocked Down PM 2', 'Sig Strikes Landed PM 2', 'Sig Strikes Attempted PM 2',
                         'Sig Strikes Absorbed PM 2', 'Strikes Landed PM 2', 'Strikes Attempted PM 2',
                         'Strikes Absorbed PM 2', 'Strike Accuracy 2', 'Takedowns PM 2', 'Takedown Attempts PM 2',
                         'Gets Taken Down PM 2', 'Submission Attempts PM 2', 'Clinch Strikes PM 2',
                         'Clinch Strikes Taken PM 2', 'Grounds Strikes PM 2', 'Ground Strikes Taken PM 2', 'Win', 'Loss'])
            all_data = list()

            # Loops through all the fight results and attempts to find the stats for each of the fighters from their
            # four prior fights. This data can then be labelled with the fight result and used to train the neural network
            # model in the 'Predictor' class, when it is called from main.
            for index, row in self.fight_results.iterrows():
                window.updateProgress(round(((index + 1) * 100) / len(self.fight_results), 2))

                try:
                    date = row[1]
                    days_since_fight = calculateDaysSince(date.split('/')[0], date.split('/')[1], date.split('/')[2])
                    name1 = row[2].rstrip()
                    name2 = row[3].rstrip()
                    result = row[4]

                    # Doesn't include any fights that happened before 2010
                    if int(date.split('/')[2]) < 2010:
                        raise Exception()

                    # Finds the stats of the two fighters prior to the date of the given fight occuring
                    fighter1_useful_data = self.findFighterStats(name1, date)
                    fighter2_useful_data = self.findFighterStats(name2, date)

                    # Produces a 'one-hot' array describing the outcome of the fight
                    if result == 2:
                        result_array = [0, 1]
                        opposite_array = [1, 0]
                    else:
                        result_array = [1, 0]
                        opposite_array = [0, 1]

                    # Concatenates the full training row with the data from both fighters and the 'one-hot' label array
                    full_list1 = fighter1_useful_data + fighter2_useful_data + result_array

                    # The training array is also reversed to help the model generalise to trends/reduce model overfitting
                    full_list2 = fighter2_useful_data + fighter1_useful_data + opposite_array

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

            training_data.to_csv('TrainingData.csv') # Saves the training dataframe to a CSV file
            self.training_data = pandas.read_csv('TrainingData.csv') # Reading from CSV ensures the expected format of the DataFrame
            print('Finished.')

        else:
            print('One or more of the necessary data files is not present. Please scrape the data using the interface button. ')
