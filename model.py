import numpy as np
import mesa
from agents import Household, Firm
import statistics

def distribute_all_profits(model):
    profits = sum(f.m_f for f in model.firms)
    if profits <= 0:
        return

    for f in model.firms:
        f.m_f = 0.0

    households = model.Households
    total_liquidity = sum(h.m_h for h in households)

    if total_liquidity <= 0:
        share = profits // len(households)
        for h in households:
            h.m_h += share
    else:
        for h in households:
            h.m_h += profits * h.m_h / total_liquidity

class LegnickModel(mesa.Model):
    """the Legnick model"""
    
    def __init__(self, n_households = 1000, n_firms = 100, seed=333, alpha = 0.9, n = 7, ld = 3, chi = 0.1, gamma = 24, delta = 0.019,
                    phi_emp_upper = 1, phi_emp_lower = 0.25, phi_price_lower = 1.025, phi_price_upper = 1.15, vartheta = 0.02, theta = 0.75,
                    psi_price = 0.25, xi = 0.01, psi_quant = 0.25, beta = 5, pie = 0.1):
        super().__init__(rng=seed)
        self.n_households = n_households
        self.n_firms = n_firms
        self.counter = 0
        self.alpha = alpha
        self.n = n
        self.ld = ld
        self.chi = chi
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.phi_emp_upper = phi_emp_upper
        self.phi_emp_lower = phi_emp_lower
        self.phi_price_lower = phi_price_lower
        self.phi_price_upper = phi_price_upper
        self.vartheta = vartheta
        self.theta = theta
        self.psi_price = psi_price
        self.xi = xi
        self.psi_quant = psi_quant
        self.pie = pie

        # data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
            "Employment": lambda m: sum(1 for h in m.agents.select(agent_type=Household) if h.type_b_connection is not None),
            "AvgPrice": lambda m: sum(f.p_f for f in m.agents.select(agent_type=Firm)) / m.n_firms,
            "AvgWage": lambda m: sum(f.w_f for f in m.agents.select(agent_type=Firm)) / m.n_firms,
            "TotalInv": lambda m: sum(f.i_f for f in m.agents.select(agent_type=Firm)),
            "PriceStd": lambda m: statistics.stdev([f.p_f for f in m.agents.select(agent_type=Firm)]),
            "WageStd": lambda m: statistics.stdev([f.w_f for f in m.agents.select(agent_type=Firm)]),
            "InvStd": lambda m: statistics.stdev([f.i_f for f in m.agents.select(agent_type=Firm)]),
            "NumOpenPositions": lambda m: sum(1 for f in m.agents.select(agent_type=Firm) if f.open_position),
            "HHLiquidity": lambda m: sum(h.m_h for h in m.agents.select(agent_type=Household)),
            "FirmLiquidity": lambda m: sum(f.m_f + f.buffer for f in m.agents.select(agent_type=Firm)),
            "UnsatisfiedDemandPct": lambda m: m.last_unsat_pct

},
        )

        # create agents
        Household.create_agents(model=self, n=n_households)
        Firm.create_agents(model=self, n=n_firms)

        self.Households = self.agents.select(agent_type = Household)
        self.firms = list(self.agents.select(agent_type = Firm))

        self.last_unsat_pct = 0.0
        self.firm_snapshots = []

        # initialise trading connectons (type a)

        for h in self.Households:
            
            type_a = self.random.sample(self.firms, 7)
            h.type_a_connections = type_a


    def step(self):
        self.counter += 1
        # collect data
        self.datacollector.collect(self)

        hh_order = list(self.Households)
        self.random.shuffle(hh_order)

        if self.counter % 21 == 1:
            #print(f"BEGINNING OF MONTH, counter={self.counter}")

            # beginning of month
            # firms:
            # each decide how to set w_f
            self.agents.select(agent_type =Firm).do("set_wages", gamma=self.gamma, delta=self.delta)
            # set employment demand and fire workers from last month
            self.agents.select(agent_type=Firm).do("set_employment_and_fire", phi_emp_upper=self.phi_emp_upper, 
                                                    phi_emp_lower=self.phi_emp_lower)

            # set prices
            self.agents.select(agent_type=Firm).do("set_prices", phi_price_upper=self.phi_price_upper,
                                                    phi_price_lower=self.phi_price_lower,
                                                    ld=self.ld,
                                                    theta=self.theta,
                                                    phi_emp_upper=self.phi_emp_upper,
                                                    vartheta=self.vartheta,
                                                    phi_emp_lower=self.phi_emp_lower)
            
            # households:
            # shuffle once, reuse the same order for every procedure this month start 
            # each search for better type_a connections
            for h in hh_order:
                h.search_connections(psi_price=self.psi_price, xi=self.xi, psi_quant=self.psi_quant)
            # job search 
            for h in hh_order:
                h.job_search(beta=self.beta, pie=self.pie)
            # decide how much m_h to spend on consumption goods
            for h in hh_order:
                h.monthly_consumption(alpha=self.alpha)
            

        # daily: 
        # households:
        # use their m_h to buy goods from random type_a connection
        # demand is equally spread thru month
        for h in hh_order:
            h.buy_goods(n=self.n)
        
        # firms produce
        self.agents.select(agent_type=Firm).do("produce", ld=self.ld)


        if self.counter % 21 == 0:
            #print(f"END OF MONTH, counter={self.counter}")
            # end of month:
            # firms:
            # use their m_f to pay wages, build buffer and pay profits
            
            self.agents.select(agent_type=Firm).do("pay_wages")
            self.agents.select(agent_type=Firm).do("add_to_buffer", chi=self.chi)
            #self.agents.select(agent_type=Firm).do("distribute_profits")
            total_unsat = sum(h.month_unsatisfied for h in self.Households)
            total_planned = sum(h.c_r_h for h in self.Households)
            self.last_unsat_pct = (total_unsat / total_planned * 100) if total_planned > 0 else 0.0

            month_num = self.counter // 21
            for f in self.firms:
                self.firm_snapshots.append({
                    'month': month_num, 'firm_id': f.unique_id,
                    'num_workers': len(f.workers), 'price': f.p_f,
                    'wage': f.w_f, 'inventory': f.i_f,
                })
            distribute_all_profits(self)


            # households:
            # adjust w_h depending on income
            self.agents.select(agent_type=Household).do("adjust_reservation_wage")




            



