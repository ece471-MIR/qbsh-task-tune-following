from data_loader import MIRQBSHDataset
from preprocessing import fill_unvoiced, compute_initial_transposition
import librosa
import numpy as np
import scipy.ndimage as ndi
from tqdm import tqdm

class DTWTFWrapper():
    database: MIRQBSHDataset = None

    # private

    def __init__(self, dataset: MIRQBSHDataset):
        self.database = dataset

    def _compute_d_beg(self, query: np.ndarray, template: np.ndarray) -> float:
        return compute_initial_transposition(query, template)

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

    def fit_template(self,
                    query_in: np.ndarray,
                    template_in: np.ndarray,
                    uni_w: bool = False,
                    bi_w: bool = False,
                    tune: bool = False,
                    fill: bool = False):
        if fill:
            template_in = fill_unvoiced(template_in)

        """
        By computing the pitch difference between the start of our query
        andf template, and then subtracting that difference from every
        element in the query, we get both sequences to start at the same key
        """
        d_beg = self._compute_d_beg(query_in, template_in)
        query_in -= d_beg

        assert (not uni_w) or (not bi_w), "Warp cannot be uni- and bi-directional!"
        if uni_w or bi_w:
            """
            subseq=True allows for open-ended DTW where Y may only be a
                susbsequence of X and does not need to reach both ends
            step_sizes_sigma=np.array([[1, 1], [0, 1]]) allows for DTW steps to
                advance both X and Y or just Y by 1. Removing the case [1, 0]
                would mean that Y (template) must always advance monotonically
                at a uniform rate, i.e. Y would not be warped. In practice,
                allowing Y to freeze is similar to allowing X to contract
                albeit more costly. DOI: 10.2478/aoa-2014-0050 does not go into
                detail about this aspect of DTW, so we parametrise it.
            
            Side Note: potential librosa.sequence.dtw bug???:
                switching X and Y but leaving asymmetric step_sizes_sigma gives
                the same results
                
                to reproduce:
                    uni_w = True (step_sizes_sigma=np.array([[1, 1], [0, 1]]))
                    case 0:
                        X = query_in, Y = template_in
                        ...
                        query_interm =  query_in[wp[:, 0]]
                        template = template_in[wp[:, 1]]
                    case 1:
                        Y = query_in, X = template_in
                        ...
                        query_interm =  query_in[wp[:, 1]]
                        template = template_in[wp[:, 0]]
                    => both cases result in same query_interm, template
                    
            """
            if uni_w:
                step_sizes_sigma = np.array([[1, 1], [0, 1]])
            else:
                step_sizes_sigma = np.array([[1, 1], [0, 1], [1, 0]])

            D, wp = librosa.sequence.dtw(
                X=query_in,
                Y=template_in,
                subseq=True,
                band_rad=0.5,
                step_sizes_sigma=step_sizes_sigma
            )

            wp = wp[::-1]
            query_interm: np.ndarray = query_in[wp[:, 0]]
            template: np.ndarray = template_in[wp[:, 1]]
        else:
            query_interm: np.ndarray = query_in
            template: np.ndarray = template_in

        if tune:
            query: np.ndarray = self._tune_follow(
                query_interm,
                template
            )
        else:
            query: np.ndarray = query_interm
        
        return query, template, query_interm

    def match_query_in_database(self,
                                query_in: np.ndarray,
                                uni_w: bool = False,
                                bi_w: bool = False,
                                tune: bool = False,
                                fill: bool = False,
                                prog_bar: bool = True) -> tuple[float, str]:
        """
            main idea is: for each point in a query, perform a dtw against every
            template to find the total "cost" in terms of distance between the
            query and template. the pair with the lowest cost is our winner

            @param query_in Interest query to compare against database
            @param uni_w Enable unidirectional dynamic time warping
            @param bi_w Enable bidirectional dynamic time warping
            @param tune Enable tune-following algorithm
            @param fill Enable left-filling unvoiced segments of templates
            @param prog_bar Toggle use of tqdm progress bar for template loop
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

            query, template, _ = self.fit_template(
                query_in=query_in,
                template_in=template,
                uni_w=uni_w,
                bi_w=bi_w,
                tune=tune,
                fill=fill
            )

            # We DGAF about the path, we grab the final cost to get there
            cost = np.sum(np.absolute(
                np.subtract(query[0:len(template)], template[0:len(query)])
            ))

            # LAZY
            costs.append(cost)
            templates.append(template_info)

        # Return the top 10 choices
        templates = np.array(templates)
        return templates[np.argsort(costs)[:10].tolist()].tolist()
