import streamlit as st
import pandas as pd
import plotly.express as px
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.tools.sql import SQLTools
from dotenv import load_dotenv, find_dotenv
import os
import json
from typing import Dict, List, Any, Optional
import re
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import pyodbc


os.environ["OPENAI_API_KEY"] =""
def extract_json(text):
    # Find content between ```json and ``` tags
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)

    if not json_match:
        raise Exception('No JSON block found')

    try:
        # Parse the extracted JSON
        json_data = json.loads(json_match.group(1))
        return json_data
    except json.JSONDecodeError:
        raise Exception('Invalid JSON format')


class InventoryVisualizer:
    def __init__(self, json_data: Dict[str, Any]):
        """
        Initialize the InventoryVisualizer with JSON data.

        Args:
            json_data (dict): Dictionary containing data and visualization suggestions
        """
        self.json_data = json_data
        self.df = pd.DataFrame(json_data["data"])
        self.viz_type = json_data.get("viz_suggestions", ["bar_chart"])[0]

        # Identify numerical and categorical columns
        self.numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns

    def create_bar_chart(self, x: Optional[str] = None, y: Optional[str] = None) -> px.bar:
        """Create a bar chart visualization."""
        x = x or self.categorical_cols[0]
        y = y or self.numerical_cols[0]

        return px.bar(
            self.df,
            x=x,
            y=y,
            title=f'{y} by {x}',
            labels={x: x.replace('_', ' ').title(),
                    y: y.replace('_', ' ').title()},
            height=500
        )

    def create_line_chart(self, x: Optional[str] = None, y: Optional[str] = None) -> px.line:
        """Create a line chart visualization."""
        x = x or self.categorical_cols[0]
        y = y or self.numerical_cols[0]
        return px.line(
            self.df,
            x=x,
            y=y,
            title=f'{y} by {x}',
            labels={x: x.replace('_', ' ').title(),
                    y: y.replace('_', ' ').title()},
            height=500
        )

    def apply_common_layout(self, fig) -> None:
        """Apply common layout settings to the figure."""
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            xaxis_title="Name",
            yaxis_title="Quantity",
            plot_bgcolor='white'
        )

    def create_visualization(self) -> Any:
        """Create visualization based on the suggested type."""
        viz_functions = {
            "Bar Chart": self.create_bar_chart,
            "bar chart": self.create_bar_chart,
            "bar_chart": self.create_bar_chart,
            "Line Chart": self.create_line_chart,
            "line chart": self.create_line_chart,
            "line_chart": self.create_line_chart
        }

        if self.viz_type not in viz_functions:
            raise ValueError(f"Unsupported visualization type: {self.viz_type}")

        fig = viz_functions[self.viz_type]()
        self.apply_common_layout(fig)
        return fig


class SQLAgentApp:
    def __init__(self):
        """Initialize the Streamlit app and the SQL Agent."""
        load_dotenv(find_dotenv())
        
        self.server = os.getenv("DB_SERVER", "antzai-tande.database.windows.net")
        self.database = os.getenv("DB_NAME", "")  
        self.username = os.getenv("DB_USER", "")        
        self.password = os.getenv("DB_PASSWORD", "")
        self.driver = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server").replace(' ', '+')
        self.db_url = f"mssql+pyodbc://{self.username}:{self.password}@{self.server}:1433/{self.database}?driver={self.driver}" #mssql+pyodbc://username:password@host:port/database?driver=ODBC+Driver+17+for+SQL+Server
        self.llm = None
        
        try:
            engine = create_engine(self.db_url)
            with engine.connect() as connection:
                print("Database connection successful!")
        except SQLAlchemyError as e:
            raise ValueError(f"Could not build the database connection: {e}")

    def select_llm(self, model_name):
        """Select the LLM dynamically based on user choice."""
        if model_name == "OpenAI":
            return OpenAIChat(id="gpt-4o-mini", temperature=0.6)
        elif model_name == "Ollama":
            return Ollama(id="llama3.2:latest")

    def create_agent(self, run_sql_query, llm):
        """Create an agent with the specified SQL query execution setting and LLM."""
        return Agent(
            name="sql agent",
            instructions=[
                "For a given user query:",
                "1. Examine the database and return the results in tabular format",
                "2. If the data is numerical and suitable for visualization, suggest appropriate chart types",
                "3. Return results in a JSON format with 'data' containing the query results and 'viz_suggestions' containing visualization recommendations"
                "5. Provide the **exact SQL query** used to fetch the data."
            ],
            tools=[
                SQLTools(
                    db_url=self.db_url,
                    list_tables=True,
                    describe_table=True,
                    run_sql_query=run_sql_query,
                    schema="local"
                )
            ],
            model=self.llm
        )

    def query_database(self, user_query, run_sql_query, llm):
        """Run the query using the agent and return the result."""
        agent = self.create_agent(run_sql_query, llm)
        response = agent.run(user_query, markdown=True)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"data": response.content, "viz_suggestions": []}

    def display_visualization(self, response_data):
        """Display visualization if visualization suggestions are present."""
        try:
            visualizer = InventoryVisualizer(response_data)
            fig = visualizer.create_visualization()
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create visualization: {str(e)}")

    def run(self):
        """Run the Streamlit UI."""
        st.title("⚡ SQL Agent with Visualization")
        st.write("🧐 Ask the agent SQL-related queries and get results with visualizations 📊")

        # Sidebar for LLM selection
        st.sidebar.header("🔮 Select LLM")
        model_name = st.sidebar.radio("Choose a model:", ("OpenAI", "Ollama"), index=1)
        self.llm = self.select_llm(model_name)

        user_query = st.text_input("💬 Enter your SQL-related question:")
        run_sql_query = st.checkbox("Enable SQL Query Execution", value=True)

        if st.button("🚀 Run Query"):
            if user_query:
                with st.spinner("⏳ Fetching results..."):
                    response = self.query_database(user_query, run_sql_query, self.llm)
                    print(response['data'])
                    json_data = extract_json(response['data'])

                    # Display visualization if available
                    if isinstance(json_data['data'], list) and json_data.get('viz_suggestions'):
                        st.subheader("📈 Visualization")
                        self.display_visualization(json_data)

                    # Display the data table
                    st.subheader("📊 Data Table")
                    if isinstance(response['data'], str):
                        st.markdown(response['data'])
                    else:
                        df = pd.DataFrame(response['data'])
                        st.dataframe(df)


if __name__ == "__main__":
    app = SQLAgentApp()
    app.run()
