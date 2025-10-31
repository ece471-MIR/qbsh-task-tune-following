class DTWWrapper():
    database: MIRQBSHDataset = None

    # private
    def __init__(self, dataset: MIRQBSHDataset):
        self.database = dataset

    def _compute_d_beg(query: list[float], template: list[float]):
        return ((query[2] - query[3]) / 2) - ((template[2] - template[3]) / 2)

    # public

    def match_query_in_database():
        # main idea is: for each point in a query, perform a dtw against every
        # template to find the total "cost" in terms of distance between the
        # query and template. the pair with the lowest cost is our winner
        pass
