table = 'fraud_detection'
stage = "@PYCONES_2022"

origin_features = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isfraud"]
features = ['TYPE_CASH_IN','TYPE_CASH_OUT','TYPE_DEBIT','TYPE_PAYMENT','TYPE_TRANSFER','AMOUNT',
'OLDBALANCEORG', 'NEWBALANCEORIG', 'OLDBALANCEDEST', 'NEWBALANCEDEST']

IMPORT_DIRECTORY_NAME = "snowflake_import_directory"

title = """
        <h1 style='text-align: center; margin-bottom: -20px;'>
        Now it's time to Machine Learning!
        </h1>
        """
subtitle = """
        <p style='text-align: center;'>
        Let's see how we can apply ML to <b>train</b> models dynamically and use them to <b>inference</b>
        """

bootstrap_help = "Whether bootstrap samples are used when\
        building trees. If False, the whole dataset is used to build each tree."

max_depth_help = "Number of samples to draw from X to train each base estimator."

criterion_help = "The function to measure the quality of a split."

criterion_options = ('gini', 'entropy', 'log_loss')