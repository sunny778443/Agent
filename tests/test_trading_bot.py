"""
Unit and Integration tests for Financial Web Search, Thinking, and Trading Bot module.
"""
import unittest

from agent.llm import LLMClient
from agent.trading_bot import (
    FinancialDatabaseClient,
    FinancialThinkingEngine,
    PortfolioManager,
    TradingBot,
    WebFinancialSearcher,
)


class TestTradingBot(unittest.TestCase):
    def setUp(self):
        self.llm = LLMClient(provider="mock")
        self.bot = TradingBot(initial_cash=100000.0, llm=self.llm)

    def test_web_financial_searcher(self):
        searcher = WebFinancialSearcher()
        news = searcher.search_financial_web("AAPL stock news", max_results=3)
        self.assertGreater(len(news), 0)
        self.assertIn("title", news[0])
        self.assertIn("snippet", news[0])

    def test_financial_database_client(self):
        db = FinancialDatabaseClient()
        aapl_data = db.query_ticker_database("AAPL")
        self.assertEqual(aapl_data["name"], "Apple Inc.")
        self.assertIn("price", aapl_data)
        self.assertIn("pe_ratio", aapl_data)

        # Unknown ticker fallback generation
        custom_data = db.query_ticker_database("CUSTOMTICKER")
        self.assertIn("CUSTOMTICKER", custom_data["name"])
        self.assertGreater(custom_data["price"], 0)

    def test_financial_thinking_engine(self):
        engine = FinancialThinkingEngine(self.llm)
        db = FinancialDatabaseClient()
        financial_data = db.query_ticker_database("NVDA")
        news = [
            {"title": "NVDA Earnings Beat Expectations", "snippet": "Strong growth and revenue beat analyst estimates", "source": "News"}
        ]

        analysis = engine.analyze_and_think("NVDA", financial_data, news, "Should I buy NVDA?")
        self.assertIn(analysis["action"], ["BUY", "SELL", "HOLD"])
        self.assertGreaterEqual(analysis["confidence"], 0.0)
        self.assertLessEqual(analysis["confidence"], 1.0)
        self.assertIn("thinking_process", analysis["reasoning"])

    def test_portfolio_manager(self):
        portfolio = PortfolioManager(initial_cash=50000.0)

        # Execute Buy
        buy_res = portfolio.execute_trade("AAPL", "BUY", price=200.0, allocation_cash=10000.0)
        self.assertEqual(buy_res["status"], "EXECUTED")
        self.assertEqual(buy_res["trade"]["shares"], 50.0)
        self.assertEqual(portfolio.cash, 40000.0)

        summary = portfolio.get_portfolio_summary({"AAPL": 210.0})
        self.assertEqual(summary["cash"], 40000.0)
        self.assertEqual(summary["holdings_value"], 10500.0)
        self.assertEqual(summary["total_equity"], 50500.0)
        self.assertEqual(summary["total_pnl"], 500.0)

        # Execute Sell
        sell_res = portfolio.execute_trade("AAPL", "SELL", price=210.0, allocation_cash=0.0)
        self.assertEqual(sell_res["status"], "EXECUTED")
        self.assertEqual(portfolio.cash, 50500.0)
        self.assertEqual(len(portfolio.positions), 0)

    def test_trading_bot_full_cycle(self):
        cycle_res = self.bot.execute_trading_cycle("What is the financial outlook for GOOGL?")
        self.assertEqual(cycle_res["search_and_thought"]["ticker"], "GOOGL")
        self.assertIn("answer", cycle_res["search_and_thought"])
        self.assertIn("trade_execution", cycle_res)
        self.assertIn("portfolio", cycle_res)
        self.assertGreater(cycle_res["portfolio"]["total_equity"], 0)


if __name__ == "__main__":
    unittest.main()
