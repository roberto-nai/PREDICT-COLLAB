import os
import json
import pandas as pd
import numpy as np
import datetime
from multiprocessing import Pool
from functools import partial

from ..constants import Task

class LogsDataProcessor:
    def __init__(self, name, filepath, columns, dir_path = "./datasets/processed", pool = 1):
        """Provides support for processing raw logs.
        Args:
            name: str: Dataset name
            filepath: str: Path to raw logs dataset
            columns: list: name of column names
            dir_path:  str: Path to directory for saving the processed dataset
            pool: Number of CPUs (processes) to be used for data processing
        """
        self._name = name
        self._filepath = filepath
        self._org_columns = columns
        self._dir_path = dir_path
        if not os.path.exists(f"{dir_path}/{self._name}/processed"):
            os.makedirs(f"{dir_path}/{self._name}/processed")
        self._dir_path = f"{self._dir_path}/{self._name}/processed"
        self._pool = pool

    def _load_df(self, sort_temporally = False):
        df = pd.read_csv(self._filepath)


        df = df[self._org_columns]
        df.columns = ["case:concept:name", 
            "concept:name", "time:timestamp"]
        df["concept:name"] = df["concept:name"].str.lower()
        df["concept:name"] = df["concept:name"].str.replace(" ", "-")
        df["time:timestamp"] = df["time:timestamp"].str.replace("/", "-")
        df["time:timestamp"]= pd.to_datetime(df["time:timestamp"],errors='coerce').map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))




        if sort_temporally:
            df.sort_values(by = ["time:timestamp"], inplace = True)
        return df

    def _extract_logs_metadata(self, df):
        keys = ["[PAD]", "[UNK]"]
        activities = list(df["concept:name"].unique())
        keys.extend(activities)
        val = range(len(keys))

        coded_activity = dict({"x_word_dict":dict(zip(keys, val))})
        code_activity_normal = dict({"y_word_dict": dict(zip(activities, range(len(activities))))})

        coded_activity.update(code_activity_normal)
        coded_json = json.dumps(coded_activity)
        with open(f"{self._dir_path}/metadata.json", "w") as metadata_file:
            metadata_file.write(coded_json)



    def _next_message_send_helper_func(self, df):
        case_id, case_name = "case:concept:name", "concept:name"
        processed_df = pd.DataFrame(columns = ["case_id",
        "prefix", "k", "next_act"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][case_name].to_list()
            for i in range(len(act) - 1):
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))
                x = True
                j=i+1
                while x:
                    next=act[j]
                    next2=next.split("_")
                    if next2[0] == "sendtask":
                        x = False
                    else:
                        if j < (len(act) - 1):
                            j = j+1
                        else:
                            next = "dummy"
                            x = False

                next_act = next
                processed_df.at[idx, "case_id"]  =  case
                processed_df.at[idx, "prefix"]  =  prefix
                processed_df.at[idx, "k"] =  i
                processed_df.at[idx, "next_act"] = next_act
                idx = idx + 1
        return processed_df

    def _process_next_message_send(self, df, train_list, test_list):
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._next_message_send_helper_func, df_split))
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/{Task.NEXT_MESSAGE_SEND.value}_train.csv", index = False)
        test_df.to_csv(f"{self._dir_path}/{Task.NEXT_MESSAGE_SEND.value}_test.csv", index = False)

    def _next_activity_helper_func(self, df):
        case_id, case_name = "case:concept:name", "concept:name"
        processed_df = pd.DataFrame(columns = ["case_id", 
        "prefix", "k", "next_act"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][case_name].to_list()
            for i in range(len(act) - 1):
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))        
                next_act = act[i+1]
                processed_df.at[idx, "case_id"]  =  case
                processed_df.at[idx, "prefix"]  =  prefix
                processed_df.at[idx, "k"] =  i
                processed_df.at[idx, "next_act"] = next_act
                idx = idx + 1

        return processed_df

    def _process_next_activity(self, df, train_list, test_list):
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._next_activity_helper_func, df_split))
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/{Task.NEXT_ACTIVITY.value}_train.csv", index = False)
        test_df.to_csv(f"{self._dir_path}/{Task.NEXT_ACTIVITY.value}_test.csv", index = False)



    def _next_time_helper_func(self, df,formatoT):
        case_id = "case:concept:name"
        event_name = "concept:name"
        event_time = "time:timestamp"
        processed_df = pd.DataFrame(columns = ["case_id", "prefix", "k", "time_passed", 
            "recent_time", "latest_time", "next_time", "remaining_time_days"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][event_name].to_list()
            time = df[df[case_id] == case][event_time].str[:19].to_list()
            time_passed = 0
            latest_diff = datetime.timedelta() #
            recent_diff = datetime.timedelta()
            next_time =  datetime.timedelta()
            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))

                if i > 0:
                    latest_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                        datetime.datetime.strptime(time[i-1], "%Y-%m-%d %H:%M:%S")
                latest_time = np.where(i == 0, 0, int((latest_diff.total_seconds() // formatoT)))

                #-----------------
                if i > 1:
                    recent_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")- \
                                    datetime.datetime.strptime(time[i-2], "%Y-%m-%d %H:%M:%S")
                recent_time = np.where(i <=1, 0, int((recent_diff.total_seconds() // formatoT)))
                #-----------

                time_passed_aux = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                  datetime.datetime.strptime(time[0], "%Y-%m-%d %H:%M:%S")
                time_passed = int((time_passed_aux.total_seconds() // formatoT))

                #--------
                if i+1 < len(time):
                    next_time = datetime.datetime.strptime(time[i+1], "%Y-%m-%d %H:%M:%S") - \
                                datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")
                    next_time = int((next_time.total_seconds() // formatoT))
                else:
                    next_time = str(1)
                #-------------

                processed_df.at[idx, "case_id"]  = case
                processed_df.at[idx, "prefix"]  =  prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "time_passed"] = time_passed
                processed_df.at[idx, "recent_time"] = recent_time
                processed_df.at[idx, "latest_time"] = latest_time
                processed_df.at[idx, "next_time"] = next_time
                idx = idx + 1
        processed_df_time = processed_df[["case_id", "prefix", "k", "time_passed", 
            "recent_time", "latest_time","next_time"]]
        return processed_df_time

    def _process_next_time(self, df, train_list, test_list,formatoT):
        df_split = np.array_split(df, self._pool)

        func_with_next_time = partial(self._next_time_helper_func, formatoT=formatoT)

        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(func_with_next_time, df_split))


        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/{Task.NEXT_TIME.value}_train.csv", index = False)
        test_df.to_csv(f"{self._dir_path}/{Task.NEXT_TIME.value}_test.csv", index = False)

    def _process_next_time_message(self, df, train_list, test_list, formatoT): # prediccion de tiempo del proximo mensaje
        df_split = np.array_split(df, self._pool)

        func_with_next_time_m = partial(self._next_time_message_helper_func, formatoT=formatoT)

        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(func_with_next_time_m, df_split))


        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/{Task.NEXT_TIME_MESSAGE.value}_train.csv", index = False)
        test_df.to_csv(f"{self._dir_path}/{Task.NEXT_TIME_MESSAGE.value}_test.csv", index = False)

    def _next_time_message_helper_func(self, df,formatoT):  # prediccion de tiempo del proximo mensaje
        case_id = "case:concept:name"
        event_name = "concept:name"
        event_time = "time:timestamp"
        processed_df = pd.DataFrame(columns=["case_id", "prefix", "k", "time_passed",
                                             "recent_time", "latest_time", "next_time", "remaining_time_days"])
        idx = 0

        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            send = df[df[case_id] == case][event_name].to_list()
            time = df[df[case_id] == case][event_time].str[:19].to_list()
            latest_diff = datetime.timedelta()
            recent_diff = datetime.timedelta()
            next_time = datetime.timedelta()

            time_message = []
            z=0

            for i in range(0, len(send)):
                prefix = np.where(i == 0, send[0], " ".join(send[:i + 1]))

                # LASTEST_TIME -------------------------
                if z > 0:
                    latest_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                  datetime.datetime.strptime(time_message[z - 1], "%Y-%m-%d %H:%M:%S")

                latest_time = np.where(z == 0 , 0, int((latest_diff.total_seconds() // formatoT)))

                # RECENT_TIME ---------------------------
                if z > 1:  # esta comparacion seria en el arreglo de "sendtask"
                    recent_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                  datetime.datetime.strptime(time_message[z - 2], "%Y-%m-%d %H:%M:%S")

                recent_time = np.where(z <= 1, 0, int((recent_diff.total_seconds() // formatoT)))

                # TIME_PASSED---------------------------

                time_passed_aux = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                              datetime.datetime.strptime(time[0], "%Y-%m-%d %H:%M:%S")

                time_passed = int((time_passed_aux.total_seconds() // formatoT))

                # Si es un nuevo send, lo agrego a la lista
                if send[i].__contains__("sendtask"):
                    time_message.insert(z, time[i])
                    z = z + 1

                # NEXT_TIME

                if i != (len(send)-1):
                    for j in range(i+1, len(send)):
                        if send[j].__contains__("sendtask"):

                            next_time = datetime.datetime.strptime(time[j], "%Y-%m-%d %H:%M:%S") - \
                                        datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")

                            next_time = int((next_time.total_seconds() // formatoT))
                            break
                        elif j == (len(send) - 1):
                            next_time = 1
                else:
                    next_time = 1



                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "time_passed"] = time_passed
                processed_df.at[idx, "recent_time"] = recent_time
                processed_df.at[idx, "latest_time"] = latest_time
                processed_df.at[idx, "next_time"] = next_time
                idx = idx + 1
        processed_df_time = processed_df[["case_id", "prefix", "k", "time_passed",
            "recent_time", "latest_time", "next_time"]]
        return processed_df_time







    def _remaining_time_helper_func(self, df,formatoT):
        case_id = "case:concept:name"
        event_name = "concept:name"
        event_time = "time:timestamp"
        processed_df = pd.DataFrame(columns = ["case_id", "prefix", "k", "time_passed", 
                "recent_time", "latest_time", "next_act", "remaining_time_days"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][event_name].to_list()
            time = df[df[case_id] == case][event_time].str[:19].to_list()
            time_passed = 0
            latest_diff = datetime.timedelta()
            recent_diff = datetime.timedelta()
            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))
                if i > 0:
                    latest_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                        datetime.datetime.strptime(time[i-1], "%Y-%m-%d %H:%M:%S")
                if i > 1:
                    recent_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")- \
                                    datetime.datetime.strptime(time[i-2], "%Y-%m-%d %H:%M:%S")

                latest_time = np.where(i == 0, 0, int((latest_diff.total_seconds() // formatoT)))
                recent_time = np.where(i <=1, 0, int((recent_diff.total_seconds() // formatoT)))

                time_passed_aux = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                  datetime.datetime.strptime(time[0], "%Y-%m-%d %H:%M:%S")
                time_passed = int((time_passed_aux.total_seconds() // formatoT))

                time_stamp = str(np.where(i == 0, time[0], time[i]))
                ttc = datetime.datetime.strptime(time[-1], "%Y-%m-%d %H:%M:%S") - \
                        datetime.datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
                ttc = str(int(ttc.total_seconds() // formatoT))

                processed_df.at[idx, "case_id"]  = case
                processed_df.at[idx, "prefix"]  =  prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "time_passed"] = time_passed
                processed_df.at[idx, "recent_time"] = recent_time
                processed_df.at[idx, "latest_time"] =  latest_time
                processed_df.at[idx, "remaining_time_days"] = ttc
                idx = idx + 1
        processed_df_remaining_time = processed_df[["case_id", "prefix", "k", 
            "time_passed", "recent_time", "latest_time","remaining_time_days"]]
        return processed_df_remaining_time

    def _process_remaining_time(self, df, train_list, test_list,formatoT):
        df_split = np.array_split(df, self._pool)

        func_remaining_time = partial(self._remaining_time_helper_func,formatoT=formatoT)

        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(func_remaining_time, df_split))


        train_remaining_time = processed_df[processed_df["case_id"].isin(train_list)]
        test_remaining_time = processed_df[processed_df["case_id"].isin(test_list)]
        train_remaining_time.to_csv(f"{self._dir_path}/{Task.REMAINING_TIME.value}_train.csv", index = False)
        test_remaining_time.to_csv(f"{self._dir_path}/{Task.REMAINING_TIME.value}_test.csv", index = False)



    def _process_remaining_time_participant(self, df, train_list, test_list, participant,formatoT):
        df_split = np.array_split(df, self._pool)

        func_with_participant = partial(self._remaining_time_participant_helper_func, participant=participant, formatoT=formatoT)

        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(func_with_participant, df_split))

        train_remaining_time = processed_df[processed_df["case_id"].isin(train_list)]
        test_remaining_time = processed_df[processed_df["case_id"].isin(test_list)]

        train_remaining_time.to_csv(f"{self._dir_path}/{Task.REMAINING_TIME_PARTICIPANT.value}_"+ participant +"_train.csv", index=False)
        test_remaining_time.to_csv(f"{self._dir_path}/{Task.REMAINING_TIME_PARTICIPANT.value}_" + participant + "_test.csv", index=False)

    def _remaining_time_participant_helper_func(self, df, participant, formatoT):
        case_id = "case:concept:name"
        event_name = "concept:name"
        event_time = "time:timestamp"
        processed_df = pd.DataFrame(columns=["case_id", "prefix", "k", "time_passed",
                                             "recent_time", "latest_time", "next_act", "remaining_time_days"])

        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            part = df[df[case_id] == case][event_name].to_list()
            time = df[df[case_id] == case][event_time].str[:19].to_list()
            latest_diff = datetime.timedelta()
            recent_diff = datetime.timedelta()

            time_participant = []
            z = 0

            for i in range(0, len(part)):
                #
                prefix = np.where(i == 0, part[0], " ".join(part[:i + 1]))

                # LASTEST_TIME -------------------------
                if z > 0:
                    latest_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                  datetime.datetime.strptime(time_participant[z - 1], "%Y-%m-%d %H:%M:%S")

                latest_time = np.where(z == 0, 0, int((latest_diff.total_seconds() // formatoT)))

                # RECENT_TIME ---------------------------
                if z > 1:  # esta comparacion seria en el arreglo de "sendtask"
                    recent_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                  datetime.datetime.strptime(time_participant[z - 2], "%Y-%m-%d %H:%M:%S")

                recent_time = np.where(z <= 1, 0, int((recent_diff.total_seconds() // formatoT)))

                # TIME_PASSED---------------------------

                time_passed_aux = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                  datetime.datetime.strptime(time[0], "%Y-%m-%d %H:%M:%S")

                time_passed = int((time_passed_aux.total_seconds() // formatoT))

                # si el participante es el elegido, agrego tiempos a arreglo auxiliar
                if part[i].__contains__(participant):
                    time_participant.insert(z, time[i])
                    z = z + 1

                # REMAINING TIME -----------
                tiempo_f = 0
                # Recorrer la lista de atrás hacia adelante
                for j in range(len(part) - 1, -1, -1):
                    if part[j].__contains__(participant):
                        tiempo_f = time[j]
                        break
                    elif time[i] == time[j]:
                        tiempo_f = 0
                        break

                if tiempo_f != 0:
                    time_stamp = str(np.where(i == 0, time[0], time[i]))
                    ttc = datetime.datetime.strptime(tiempo_f, "%Y-%m-%d %H:%M:%S") - \
                          datetime.datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
                    ttc = str(int(ttc.total_seconds() // formatoT))
                else:
                    ttc = str(0)

                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "time_passed"] = time_passed
                processed_df.at[idx, "recent_time"] = recent_time
                processed_df.at[idx, "latest_time"] = latest_time
                processed_df.at[idx, "remaining_time_days"] = ttc
                idx = idx + 1
        processed_df_remaining_time = processed_df[["case_id", "prefix", "k",
                                                    "time_passed", "recent_time", "latest_time", "remaining_time_days"]]
        return processed_df_remaining_time


    def process_logs(self, task, participant,formatoT,
        sort_temporally = False, 
        train_test_ratio = 0.80):
        df = self._load_df(sort_temporally)
        self._extract_logs_metadata(df)
        train_test_ratio = int(abs(df["case:concept:name"].nunique()*train_test_ratio))
        train_list = df["case:concept:name"].unique()[:train_test_ratio]
        test_list = df["case:concept:name"].unique()[train_test_ratio:]
        if task == Task.NEXT_ACTIVITY:
            self._process_next_activity(df, train_list, test_list)
        elif task == Task.NEXT_TIME:
            self._process_next_time(df, train_list, test_list,formatoT)
        elif task == Task.REMAINING_TIME:
            self._process_remaining_time(df, train_list, test_list,formatoT)
        elif task == Task.NEXT_MESSAGE_SEND:
            self._process_next_message_send(df, train_list, test_list)
        elif task == Task.NEXT_TIME_MESSAGE:
            self._process_next_time_message(df, train_list, test_list,formatoT)
        elif task == Task.REMAINING_TIME_PARTICIPANT:
            self._process_remaining_time_participant(df, train_list, test_list, participant,formatoT)
        else:
            raise ValueError("Invalid task.")