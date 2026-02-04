import hashlib
from typing import List, Dict, Any, Callable, Optional
import pandas as pd
from backtest.strategy_registry import resolve_callable

class Model:
    def __init__(self,id:int, name: str,type :str, label:str,label_base:str,predict_col:str, proba_col:str,
                  filename:str, features: Optional[List[str]]=None,
                    sk_dataset: Optional[int] = None, sk_label: Optional[int] = None, sk_symbol: Optional[int] = None):
        """Initialize the Model.

        Args:
            id (int): The unique identifier for the model.
            name (str): The name of the model.
            type (str): The type of the model.
            label (str): The label for the model.
            label_base (str): The base label for the model.
            predict_col (str): The column name for the predicted values.
            proba_col (str): The column name for the predicted probabilities.
            filename (str): The filename for saving the model.
            features (List[str]): The list of features used by the model.
            sk_dataset (int): The foreign key to the dataset.
            sk_label (int): The foreign key to the label.
            sk_symbol (int): The foreign key to the symbol.
        """
        self.id = id
        self.name = name
        self.type = type
        self.label = label
        self.label_base = label_base
        self.predict_col = predict_col
        self.proba_col = proba_col
        self.filename = filename
        self.features = features
        self.sk_dataset = sk_dataset
        self.sk_label=sk_label
        self.sk_symbol=sk_symbol

    def to_dict(self) -> Dict[str, Any]:
        """
        Map Model instance to a dict matching the MODEL ORM attribute names.
        Fields not present on the Model instance are set to None.
        """
        return {
            "sk_dataset": getattr(self, "sk_dataset", None),
            "sk_symbol": getattr(self, "sk_symbol", None),
            "sk_label": getattr(self, "sk_label", None),
            "algo": getattr(self, "algo", None),
            "name": self.name,
            "comment": None ,
            "header_dts": None if self.features is None else str(self.features),
            "file_name": self.filename,
            "type_model": self.type,
            "lib_label": self.label if isinstance(self.label, str) else None,
            "lib_predict_label": self.predict_col,
            "lib_proba_label": self.proba_col,
        }

class StrategyType:
    def __init__(
        self,
        id: int,
        name: str,
        # model_type: str,
        description: str,
        code_entry: str = "",
        code_exit: str = "",
        param_entry: str = "",
        param_exit: str = "",
    ):
        """Initialize the StrategyType.

        Args:
            id (int): The unique identifier for the strategy type.
            name (str): The name of the strategy type.
            # model_type (str): The type of model used in the strategy.
            description (str): A description of the strategy type.
            code_entry (str, optional): The code for the entry condition. Defaults to None.
            code_exit (str, optional): The code for the exit condition. Defaults to None.
            param_entry (str, optional): The parameters for the entry condition. Defaults to None.
            param_exit (str, optional): The parameters for the exit condition. Defaults to None.
        """
        self.id = id
        self.name = name
        # self.model_type = model_type
        self.description = description
        self.code_entry = code_entry or ""
        self.code_exit = code_exit or ""
        self.param_entry = param_entry or ""
        self.param_exit = param_exit or ""

    def calculate_id(self, models: List[Model], num_strat: int=0, setting_id: int=0) -> int:
        """Calculate a calculated ID for a strategy.

        Args:
            models (List[Model]): The list of models used in the strategy.
            num_strat (int): The number of strategies.
            setting_id (int): The ID of the settings used in the strategy.

        Returns:
            int: A calculated ID.
        """
        my_id = self.id * 1000000000
        for m in models:
            my_id += m.id * (10 ** (len(models) - models.index(m)+2 ))
        my_id += num_strat*10
        my_id += setting_id

        return my_id

    def safe_resolve_callable(self, path: str | None, label: str) -> Callable:
        if not path:
            raise ValueError(f"{label} is missing in StrategyType '{self.name}'")
        return resolve_callable(path)
    
    def get_code_entry(self) -> Callable:
        """Get the entry condition code as a callable function.

        Returns:
            Callable: The entry condition function.
        """
        return self.safe_resolve_callable(self.code_entry, "code_entry")
    
    def get_code_exit(self) -> Callable:
        """Get the exit condition code as a callable function.

        Returns:
            Callable: The exit condition function.
        """
        return self.safe_resolve_callable(self.code_exit, "code_exit")

# class strategy : a set of models with entry and exit conditions
class Strategy:
    def __init__(self,id:int, name: str, type:StrategyType, models: List[Model],
                  entry_condition: Callable, exit_condition: Callable, param_entry: dict[str, Any],param_exit: dict[str, Any],
                  suffix:str=""):
        """Initialize the Strategy.

        Args:
            id (int): The unique identifier for the strategy.
            name (str): The name of the strategy.
            type (StrategyType): The type of strategy.
            models (List[Model]): The list of models used in the strategy.
            entry_condition (Callable): The function defining the entry condition.
            exit_condition (Callable): The function defining the exit condition.
            param_entry (dict[str, Any]): The parameters for the entry condition.
            param_exit (dict[str, Any]): The parameters for the exit condition.
            suffix (str, optional): Suffix for entry and exit signals. Defaults to "".
        """
        self.id = id
        self.name = name
        self.type = type
        self.models = models
        self.entry_condition = entry_condition
        self.exit_condition = exit_condition
        self.param_entry = param_entry or {}
        self.param_exit = param_exit or {}
        self.suffix = suffix

        if self.suffix == "":
            self.suffix = str(self.id)

    def to_dict(self):
        return {
            "name": self.name,
            "strat_type": self.type.name if self.type else None,
            # "model_type": self.type.model_type if self.type else None,
            "description": self.type.description if self.type else None,
            "param_entry": str(self.param_entry),  # Convert dict to string for DB storage
            "param_exit": str(self.param_exit),  # Convert dict to string for DB storage
        }

        # calculate the id, init the strategy and return it
    @classmethod
    def create(
        cls,
        type: StrategyType,
        models: List[Model],
        entry_condition: Callable,
        exit_condition: Callable,
        param_entry: Dict[str, Any],
        param_exit: Dict[str, Any],
        num_strat: int = 0,
    ) -> "Strategy":
        """Create a Strategy instance with a calculated ID.

        Args:
            type (StrategyType): The type of strategy.
            models (List[Model]): The list of models used in the strategy.
            num_strat (int): The number of strategies.

        Returns:
            Strategy: The created Strategy instance.
        """
        strat_id = type.calculate_id(models, num_strat)
        return cls(
            id=strat_id,
            name=f"Strategy {strat_id}",
            type=type,
            models=models,
            entry_condition=entry_condition,
            exit_condition=exit_condition,
            param_entry=param_entry,
            param_exit=param_exit,
        )

    def add_signals(self, df: pd.DataFrame, default_value: int=0,
                    sort_column: str="",sort_group_column: str="") -> pd.DataFrame:
        """Add entry and exit signals to the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            default_value (int, optional): The default value for signals. Defaults to 0.
            sort_column (str, optional): The column to sort by. Defaults to "". Put in settings ? 
            sort_group_column (str, optional): The column to group by for ranking. Defaults to "". Put in settings ? 

        Returns:
            pd.DataFrame: The DataFrame with added entry and exit signals.
        """

        df[f'entry_{self.suffix}'] = self.entry_condition(df,self.models,self.param_entry).fillna(default_value)
        df[f'exit_{self.suffix}'] = self.exit_condition(df,self.models,self.param_exit).fillna(default_value)

        if sort_column:
            df['tmp_filter'] = df[sort_column].where(df[f'entry_{self.suffix}'] == 1, 0.0)
            df[f'rank_{self.suffix}'] = df.groupby(sort_group_column)[sort_column].rank(ascending=False, method='first')

        df.drop(columns=['tmp_filter'], errors='ignore', inplace=True)

        return df


# class scenario : a strategy with a money management and options for backtesting
class Scenario:
    def __init__(self, id: int, strategy: Strategy, stop_loss: str, # Callable, #pos_size: Callable,
                  params: Optional[Dict[str, Any]] = None):
        """Initialize the Scenario.

        Args:
            id (int): The unique identifier for the scenario.
            strategy (Strategy): The strategy to be used in the scenario.
            stop_loss (str): The function defining the stop loss.
            #pos_size (Callable): The function defining the position size.
            params (Dict[str, Any], optional): The parameters for the scenario. Defaults to None.
            e.g. options = {'max_positions': 5, 'position_size': 1000, 'scale_up': False, 'sell_all': True, 'fixe_quantity':False}
        """
        self.id = id
        self.strategy = strategy
        self.stop_loss = stop_loss
        self.params = params or {}
        self.sk_campaign = None
        self.code= None

    def to_dict(self):
        return {
            "sk_strategy": self.strategy.id,
            "settings": str(self.params),  # Convert dict to string for DB storage
            "comment": f"Scenario for strategy {self.strategy.name}",
            "sk_campaign": self.sk_campaign,
            "code": self.code
        }
    
    def set_sk_campaign(self, sk_campaign: int)-> str:
        """Set the sk_campaign and generate the code for the scenario.
        Args:
            sk_campaign (int): The sk_campaign to be set.
        Returns:
            str: The generated code for the scenario.
        """
        self.sk_campaign = sk_campaign
        code_hash = hashlib.md5(str(self.params).encode()).hexdigest()[:10]  # Simple hash to keep code manageable
        self.code = f"{self.sk_campaign}_{self.strategy.id}_{code_hash}"
        return self.code

    def add_sl(self, df: pd.DataFrame, col: str, sl: float) -> pd.DataFrame:
        """Add stop loss signals to the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            col (str): The column to apply the stop loss to.
            sl (float): The stop loss percentage.

        Returns:
            pd.DataFrame: The DataFrame with added stop loss signals.
        """
        df_tmp = df.copy()
        df_tmp.loc[:, f'sl_{self.strategy.suffix}'] = df_tmp[col] * (1 - sl)
        return df_tmp

    def run(self, df: pd.DataFrame):
        raise NotImplementedError("Scenario.run must be implemented by the caller")



class Campaign:
    def __init__(self, name: str, description: str, scenarii: List[Scenario], params: Optional[Dict[str, Any]] = None):
        """Initialize the Campaign.

        Args:
            name (str): The name of the campaign.
            description (str): A description of the campaign.
            scenarii (List[Scenario]): The list of scenarii used in the campaign.
            params (Dict[str, Any], optional): The parameters for the campaign. Defaults to None.
        """
        self.name = name
        self.description = description
        self.scenarii = scenarii
        self.params = params or {}
        self.filename = f"{self.name.lower().replace(' ', '_')}_campaign.txt"
        self.id: Optional[int] = None
        self.code = self.name.lower().replace(" ", "_")


    def to_dict(self):
        return {
            "code": self.code,
            "description": self.description,
            "settings": str(self.params),  # Convert dict to string for DB storage
            "filename": self.filename,
            "initial_cash": self.params.get("initial_cash", None),
            "commission": self.params.get("commission", None),
            "date_start": str(self.params.get("date_start", None)),
            "date_end": str(self.params.get("date_end", None)),

        }

    def run(self, df):
        results = {}
        for scenario in self.scenarii:
            results[scenario.id] = scenario.run(df.copy())
        return results

# Example usage (to be replaced with real logic)
if __name__ == "__main__":
    # Dummy predict function
    def dummy_predict(df):
        return [0] * len(df)

    # Dummy entry/exit conditions
    def entry_cond(df, models, settings):
        return df.index % 2 == 0

    def exit_cond(df, models, settings):
        return df.index % 2 == 1

    model1 = Model(
        id=1,
        name="DummyModel",
        type="dummy",
        label="label",
        label_base="label_base",
        predict_col="predict",
        proba_col="proba",
        filename="dummy_model.pkl",
        features=[],
    )
    strat_type = StrategyType(id=1, name="DummyType",  description="Dummy strategy type")
    strat1 = Strategy(
        id=1,
        name="Strat1",
        type=strat_type,
        models=[model1],
        entry_condition=entry_cond,
        exit_condition=exit_cond,
        param_entry={},
        param_exit={},
    )
    scenario1 = Scenario(id=1, strategy=strat1, stop_loss="fixed", params={})
    campaign = Campaign(name="TestCampaign", description="Test campaign", scenarii=[scenario1])

    import pandas as pd
    df = pd.DataFrame({'A': range(10)})