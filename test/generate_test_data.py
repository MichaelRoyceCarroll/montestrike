#!/usr/bin/env python3
"""
MonteStrike Test Data Generator

Generates realistic option scenario data using YFinance API for testing the MonteStrike library.
Follows caching patterns from iwm_scanner.py with structured JSON output.

Usage:
    python generate_test_data.py
    python generate_test_data.py --symbol IWM --min-days 4 --max-days 10
    python generate_test_data.py --symbol U --min-days 10 --max-days 30 --option-type put
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import glob
import argparse
import sys
from typing import Dict, List, Optional, Tuple

class TestDataGenerator:
    """Generate test data for MonteStrike library using YFinance API"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cleanup_old_cache()
    
    def cleanup_old_cache(self, days_to_keep: int = 7):
        """Remove cache files older than specified days"""
        cutoff_date = datetime.now().date() - timedelta(days=days_to_keep)
        
        cache_files = glob.glob(os.path.join(self.cache_dir, "*_data_*.json"))
        for cache_file in cache_files:
            try:
                # Extract date from filename pattern: symbol_data_YYYY-MM-DD.json
                basename = os.path.basename(cache_file)
                if "_data_" in basename:
                    date_part = basename.split("_data_")[1].replace(".json", "")
                    file_date = datetime.strptime(date_part, "%Y-%m-%d").date()
                    
                    if file_date < cutoff_date:
                        os.remove(cache_file)
                        print(f"🧹 Cleaned up old cache file: {basename}")
            except (ValueError, OSError) as e:
                print(f"⚠️  Could not process cache file {cache_file}: {e}")
    
    def get_cached_data(self, symbol: str) -> Optional[Dict]:
        """Check for cached market data for today"""
        today = datetime.now().date()
        cache_filename = f"{symbol.lower()}_data_{today}.json"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    print(f"📁 Using cached data for {symbol}")
                    return data
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return None
    
    def save_cached_data(self, symbol: str, data: Dict):
        """Save market data to cache"""
        today = datetime.now().date()
        cache_filename = f"{symbol.lower()}_data_{today}.json"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Cached data for {symbol}")
        except Exception as e:
            print(f"⚠️  Could not save cache for {symbol}: {e}")
    
    def calculate_historical_metrics(self, symbol: str, period: str = "30d") -> Tuple[float, float]:
        """Calculate drift (μ) and volatility (σ) from historical data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                print(f"⚠️  No historical data for {symbol}, using defaults")
                return 0.05, 0.20  # Default 5% drift, 20% volatility
            
            # Calculate daily returns
            returns = hist['Close'].pct_change().dropna()
            
            if len(returns) < 10:
                print(f"⚠️  Insufficient historical data for {symbol}, using defaults")
                return 0.05, 0.20
            
            # Annualize metrics
            trading_days_per_year = 252
            drift = returns.mean() * trading_days_per_year
            volatility = returns.std() * np.sqrt(trading_days_per_year)
            
            # Sanity checks
            if not np.isfinite(drift) or abs(drift) > 1.0:
                drift = 0.05
            if not np.isfinite(volatility) or volatility <= 0 or volatility > 2.0:
                volatility = 0.20
            
            print(f"📊 {symbol} calculated metrics: drift={drift:.4f}, volatility={volatility:.4f}")
            return float(drift), float(volatility)
            
        except Exception as e:
            print(f"⚠️  Error calculating metrics for {symbol}: {e}")
            return 0.05, 0.20
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if hist.empty:
                return None
            return float(hist['Close'].iloc[-1])
        except Exception as e:
            print(f"❌ Error fetching current price for {symbol}: {e}")
            return None
    
    def filter_options_by_expiration(self, option_chain, min_days: int, max_days: int) -> List[Tuple[str, int]]:
        """Filter option expirations by date range"""
        today = datetime.now().date()
        filtered_expirations = []
        
        for exp_str in option_chain:
            try:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                days_out = (exp_date - today).days
                
                if min_days <= days_out <= max_days:
                    filtered_expirations.append((exp_str, days_out))
            except ValueError:
                continue
        
        return sorted(filtered_expirations, key=lambda x: x[1])
    
    def generate_option_scenarios(self, symbol: str, min_days: int, max_days: int, 
                                option_type: str = "call") -> List[Dict]:
        """Generate test scenarios for a symbol and date range"""
        
        # Check cache first
        cached_data = self.get_cached_data(symbol)
        if cached_data and cached_data.get('min_days') == min_days and cached_data.get('max_days') == max_days:
            return cached_data.get('scenarios', [])
        
        print(f"🌐 Fetching live data for {symbol} ({min_days}-{max_days} days, {option_type}s)")
        
        # Get current price
        current_price = self.get_current_price(symbol)
        if not current_price:
            print(f"❌ Could not fetch current price for {symbol}")
            return []
        
        print(f"💰 Current {symbol} price: ${current_price:.2f}")
        
        # Calculate historical metrics
        drift, volatility = self.calculate_historical_metrics(symbol)
        
        # Get option chain
        try:
            ticker = yf.Ticker(symbol)
            all_expirations = ticker.options
            
            if not all_expirations:
                print(f"❌ No options data available for {symbol}")
                return []
            
            print(f"✅ Found {len(all_expirations)} available expirations")
            
        except Exception as e:
            print(f"❌ Error fetching options for {symbol}: {e}")
            return []
        
        # Filter expirations
        filtered_exps = self.filter_options_by_expiration(all_expirations, min_days, max_days)
        
        if not filtered_exps:
            print(f"❌ No {option_type} options found in {min_days}-{max_days} day range")
            return []
        
        print(f"🎯 Found {len(filtered_exps)} expirations in target range")
        
        scenarios = []
        
        # Process each expiration
        for exp_date, days_out in filtered_exps:
            try:
                option_chain = ticker.option_chain(exp_date)
                options_data = option_chain.calls if option_type == "call" else option_chain.puts
                
                if options_data.empty:
                    continue
                
                # Filter options near current price (within 20%)
                price_range = current_price * 0.20
                min_strike = current_price - price_range
                max_strike = current_price + price_range
                
                filtered_options = options_data[
                    (options_data['strike'] >= min_strike) & 
                    (options_data['strike'] <= max_strike)
                ]
                
                if filtered_options.empty:
                    continue
                
                print(f"📋 Processing {exp_date}: {len(filtered_options)} {option_type}s near current price")
                
                # Generate scenarios for each suitable strike
                for _, option in filtered_options.head(5).iterrows():  # Limit to 5 per expiration
                    scenario = {
                        "symbol": symbol,
                        "option_type": option_type,
                        "expiration_date": exp_date,
                        "days_to_expiration": days_out,
                        "current_price": current_price,
                        "strike_price": float(option['strike']),
                        "time_to_expiration": days_out / 365.0,
                        "drift": drift,
                        "volatility": volatility,
                        "steps_per_day": 50,
                        "path_counts": [50000, 100000, 500000, 1000000, 2000000, 4000000],
                        "option_market_data": {
                            "last_price": float(option['lastPrice']) if pd.notna(option['lastPrice']) else 0.0,
                            "bid": float(option['bid']) if pd.notna(option['bid']) else 0.0,
                            "ask": float(option['ask']) if pd.notna(option['ask']) else 0.0,
                            "volume": int(option['volume']) if pd.notna(option['volume']) else 0,
                            "implied_volatility": float(option['impliedVolatility']) if pd.notna(option['impliedVolatility']) else volatility
                        },
                        "generated_timestamp": datetime.now().isoformat(),
                        "moneyness": (option['strike'] / current_price - 1) * 100
                    }
                    scenarios.append(scenario)
                    
            except Exception as e:
                print(f"⚠️  Error processing expiration {exp_date}: {e}")
                continue
        
        # Cache the results
        cache_data = {
            "symbol": symbol,
            "option_type": option_type,
            "min_days": min_days,
            "max_days": max_days,
            "generated_date": datetime.now().date().isoformat(),
            "scenarios": scenarios
        }
        self.save_cached_data(symbol, cache_data)
        
        print(f"✅ Generated {len(scenarios)} test scenarios for {symbol}")
        return scenarios
    
    def save_test_data(self, scenarios: List[Dict], output_file: str):
        """Save test scenarios to JSON file"""
        output_path = os.path.join("data/test_data", output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        output_data = {
            "metadata": {
                "generated_timestamp": datetime.now().isoformat(),
                "total_scenarios": len(scenarios),
                "generator_version": "1.0.0",
                "disclaimer": "This data is for testing purposes only and does not constitute financial advice."
            },
            "scenarios": scenarios
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"💾 Saved {len(scenarios)} scenarios to {output_path}")
        except Exception as e:
            print(f"❌ Error saving test data: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate MonteStrike test data")
    parser.add_argument("--symbol", default="IWM", help="Stock symbol (default: IWM)")
    parser.add_argument("--min-days", type=int, default=4, help="Minimum days to expiration (default: 4)")
    parser.add_argument("--max-days", type=int, default=10, help="Maximum days to expiration (default: 10)")
    parser.add_argument("--option-type", choices=["call", "put"], default="call", help="Option type (default: call)")
    parser.add_argument("--output", help="Output filename (auto-generated if not specified)")
    
    args = parser.parse_args()
    
    print("🔍 MonteStrike Test Data Generator")
    print(f"📅 Generating {args.option_type} data for {args.symbol} ({args.min_days}-{args.max_days} days)")
    print("=" * 60)
    
    generator = TestDataGenerator()
    
    try:
        scenarios = generator.generate_option_scenarios(
            args.symbol, args.min_days, args.max_days, args.option_type
        )
        
        if not scenarios:
            print("❌ No scenarios generated")
            sys.exit(1)
        
        # Auto-generate filename if not specified
        if not args.output:
            args.output = f"{args.symbol.lower()}_{args.option_type}_{args.min_days}-{args.max_days}d.json"
        
        generator.save_test_data(scenarios, args.output)
        
        print(f"\n✅ Test data generation complete!")
        print(f"📊 Generated {len(scenarios)} scenarios")
        print(f"💾 Saved to data/test_data/{args.output}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()