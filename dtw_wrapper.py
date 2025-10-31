from data_loader import MIRQBSHDataset
import librosa
import numpy as np
from tqdm import tqdm

class DTWWrapper():
    database: MIRQBSHDataset = None

    # private

    def __init__(self, dataset: MIRQBSHDataset):
        self.database = dataset


    def _compute_d_beg(self, query: list[float], template: list[float]):
        return ((query[2] - query[3]) / 2) - ((template[2] - template[3]) / 2)

    def _tune_follow(self,
                     query: np.ndarray,
                     template: np.ndarray) -> np.ndarray:
        """
            By computing the pitch difference between the start of our query
            andf template, and then subtracting that difference from every
            element in the query, we get both sequences to start at the same key
        """
        d_beg = self._compute_d_beg(query, template)
        query -= d_beg
        return query

    # public

    def match_query_in_database(self,
                                query_in: np.ndarray,
                                tuned: bool = False) -> tuple[float, str]:
        """
            main idea is: for each point in a query, perform a dtw against every
            template to find the total "cost" in terms of distance between the
            query and template. the pair with the lowest cost is our winner

            @param query_in Interest query to compare against database
            @param tuned Enable tune-following algorithm
        """

        costs: list[np.float64] = []
        templates: list[str] = []
        for template_info in tqdm(self.database.song_list):
            template = self.database.load_template_midi(
                template_info
            )
            if tuned:
                query: np.ndarray = self._tune_follow(query_in, template)
            else:
                query: np.ndarray = query_in

            # We DGAF about the path, we grab the final cost to get there
            cost = librosa.sequence.dtw(X=query, Y=template)[0][-1,-1]

            # LAZY
            costs.append(cost)
            templates.append(template_info)

        # Return the most confident choice
        return templates[np.argmin(costs)]
