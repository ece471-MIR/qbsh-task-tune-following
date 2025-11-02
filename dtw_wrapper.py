from data_loader import MIRQBSHDataset
from preprocessing import fill_unvoiced
import librosa
import numpy as np
import scipy.ndimage as ndi
from tqdm import tqdm

class DTWWrapper():
    database: MIRQBSHDataset = None

    # private

    def __init__(self, dataset: MIRQBSHDataset):
        self.database = dataset


    def _compute_d_beg(self, query: list[float], template: list[float]):
        return ((query[2] + query[3]) / 2) - ((template[2] + template[3]) / 2)

    def _tune_follow(self,
                     query: np.ndarray,
                     template: np.ndarray,
                     align_speed: float = 0.05) -> np.ndarray:
        """
            see figure 6 in DOI: 10.2478/aoa-2014-0050
            credit to Bartlomiej Stasiak

            we are recursively calculating filter coeffs to scale our query
            vector by. this is just taking the error at every time step and
            biasing it with the previous time step to tell us whether to pitch
            down or pitch up.

            the rate of change can be adjusted (`align_speed`)
        """
        # recursively calculate tune following
        q_len = min(map(len, [query, template]))
        query = np.resize(query, (q_len,))
        template = np.resize(template, (q_len,))

        # preallocate array to hold the query w/ tune following
        query_wtf = np.empty([q_len,], dtype=np.float64)*np.nan

        query_wtf[0] = query[0]
        e_i = align_speed * (-query[0] + template[0])
        for idx in range(1, q_len):
            """
            For each step:
                calculate the error between the template and query
                feed back the previous time step's error, scaled by how
                aggressively we want to match the pitch
            """
            query_wtf[idx] = query[idx] + e_i
            e_i = align_speed * (-query[idx] + template[idx]
                + (((1-align_speed) / align_speed) * e_i))
        
        return query_wtf

    # public

    def match_query_in_database(self,
                                query_in: np.ndarray,
                                warp: bool = False,
                                tuned: bool = False,
                                fill_temp: bool = False,
                                prog_bar: bool = True) -> tuple[float, str]:
        """
            main idea is: for each point in a query, perform a dtw against every
            template to find the total "cost" in terms of distance between the
            query and template. the pair with the lowest cost is our winner

            @param query_in Interest query to compare against database
            @param tuned Enable tune-following algorithm
        """

        costs: list[np.float64] = []
        templates: list[str] = []

        if prog_bar:
            template_iter = tqdm(self.database.song_list)
        else:
            template_iter = self.database.song_list

        for template_info in template_iter:
            template = self.database.load_template_midi(
                template_info
            )

            if fill_temp:
                template = fill_unvoiced(template)

            """
            By computing the pitch difference between the start of our query
            andf template, and then subtracting that difference from every
            element in the query, we get both sequences to start at the same key
            """
            d_beg = self._compute_d_beg(query_in, template)
            query_in -= d_beg

            if warp:
                D, wp = librosa.sequence.dtw(
                    Y=query_in,
                    X=template[0:len(query_in)],
                    band_rad=0.5
                )
                query_interm: np.ndarray = query_in[wp[:,0][::-1]]
            else:
                query_interm: np.ndarray = query_in

            if tuned:
                query: np.ndarray = self._tune_follow(
                    query_interm,
                    template
                )
            else:
                query: np.ndarray = query_interm

            # We DGAF about the path, we grab the final cost to get there
            cost = np.sum(np.absolute(query[0:len(template)] - template[0:len(query)]))

            # LAZY
            costs.append(cost)
            templates.append(template_info)

        # Return the top 10 choices
        templates = np.array(templates)
        return templates[np.argsort(costs)[:10].tolist()].tolist()
