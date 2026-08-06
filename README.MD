# 📊 Monthly Loan Portfolio Roll-Rate & Vintage Analysis Engine

> **Production-Grade Collections & Loss Forecasting Framework**  
> Quantitative Collections Analytics | 100,000-Loan Panel Dataset | 24-Month Vintage Cohorts | Markov Chain Forecasting

---

## 🎯 Executive Summary

This flagship project constructs a **production-grade monthly portfolio performance engine** simulating 100,000 retail and SME loans across 24 origination cohorts and 31 monthly snapshots (Jan 2024 → Jul 2026). The engine tracks **3.1M loan-month observations**, computes **vintage loss curves by Months on Books (MOB)**, builds a **discrete-time Markov Chain transition matrix** for delinquency state dynamics, **projects portfolio losses** using matrix power multiplication across 3-, 6-, and 12-month operational horizons, and trains a **predictive model to optimize collections interventions**.

Regulatory compliance and collections operations teams use this framework to:
- **Monitor cohort decay** independent of portfolio growth
- **Forecast delinquency escalation** via state transition forecasting
- **Optimize collections targeting** based on cure probability predictions
- **Stress-test portfolios** across macroeconomic scenarios

---

## 📋 Table of Contents

1. [Domain Context & Problem Statement](#domain-context--problem-statement)
2. [Architecture & Methodology](#architecture--methodology)
3. [Synthetic Data Generation](#synthetic-data-generation)
4. [Vintage Loss Curve Analysis](#vintage-loss-curve-analysis)
5. [Markov Chain Transition Matrix](#markov-chain-transition-matrix)
6. [Portfolio Loss Forecasting](#portfolio-loss-forecasting)
7. [Collections Optimization Model](#collections-optimization-model)
8. [Dashboard Visualizations](#dashboard-visualizations)
9. [Key Findings](#key-findings)
10. [Model Performance Metrics](#model-performance-metrics)
11. [Files & Directory Structure](#files--directory-structure)
12. [How to Run](#how-to-run)
13. [Portfolio Context](#portfolio-context)

---

## Domain Context & Problem Statement

### Credit Risk Operations Challenge
Collections teams and credit risk functions operate under several constraints:

1. **Cohort-Specific Decay:** Portfolio growth masks underlying cohort performance degradation. A 10% delinquency rate in a growing portfolio may reflect improving underwriting or deteriorating seasoning.
2. **Delinquency Dynamics:** Accounts transition between DPD states monthly—some roll forward to worse buckets, others cure back to current. Aggregate metrics miss these flows.
3. **Intervention Optimization:** Collections resources are limited. Targeting decisions require predictive estimates of which accounts are likely to cure if contacted.
4. **Operational Forecasting:** Risk committees need 3-, 6-, and 12-month projections to adjust collection intensity and provisioning policies.

### This Engine Solves For
✅ **Vintage-Isolated Performance Tracking** via MOB-indexed cohort curves  
✅ **State Flow Analytics** via Markov transition matrices quantifying roll-forward and cure rates  
✅ **Stochastic Forecasting** via matrix power multiplication (v_n = v_0 · P^n)  
✅ **Collections Targeting** via logistic regression cure probability models  
✅ **Regulatory Documentation** via production-grade 6-panel executive dashboard  

---