import numpy as np
import mesa
import random
from agents import Household, Firm
import statistics

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
            "PositionsStd": lambda m: statistics.stdev(f.n_positions for f in m.agents.select(agent_type=Firm)),
            "AvgPositions": lambda m: sum(f.n_positions for f in m.agents.select(agent_type=Firm)) / m.n_firms

}
        )

        # create agents
        Household.create_agents(model=self, n=n_households)
        Firm.create_agents(model=self, n=n_firms)

        Households = self.agents.select(agent_type = Household)
        firms = list(self.agents.select(agent_type = Firm))

        # initialise trading connectons (type a)

        for h in Households:
            
            type_a = self.random.sample(firms, 7)
            h.type_a_connections = type_a
        
        # initialise type b connections
        household_list = list(Households)
        random.shuffle(household_list)
        for i, f in enumerate(firms):
            for h in household_list[i * 10 : (i + 1) * 10]:
                h.type_b_connection = f
                f.workers.append(h)
            # give firm enough capital to pay first months wages
            f.m_f = f.w_f * len(f.workers)
            f.i_f = self.ld * len(f.workers) * 21
            f.demand = (n_households / n_firms) * (100.0 ** alpha)  # expected monthly demand ≈ 631

    def step(self):
        self.counter += 1
        # collect data
        self.datacollector.collect(self)

        if self.counter % 21 == 1:
            print(f"BEGINNING OF MONTH, counter={self.counter}")

            # beginning of month
            # firms:
            # each decide how to set w_f
            self.agents.select(agent_type =Firm).do("set_wages", gamma=self.gamma, delta=self.delta)
            # fire workers from last month
            self.agents.select(agent_type=Firm).do("fire_workers")
            # set employment demand
            self.agents.select(agent_type=Firm).do("set_employment", phi_emp_upper=self.phi_emp_upper,
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
            # each search for better type_a connections
            self.agents.select(agent_type=Household).shuffle_do("search_connections", psi_price=self.psi_price, xi=self.xi, psi_quant=self.psi_quant)
            # job search 
            self.agents.select(agent_type=Household).do("job_search", beta=self.beta, pie=self.pie)
            # decide how much m_h to spend on consumption goods
            self.agents.select(agent_type=Household).do("monthly_consumption", alpha=self.alpha)
            

        # daily: 
        # households:
        # use their m_h to buy goods from random type_a connection
        # demand is equally spread thru month
        
        self.agents.select(agent_type=Household).shuffle_do("buy_goods", n=self.n)
        
        # firms produce
        self.agents.select(agent_type=Firm).do("produce", ld=self.ld)


        if self.counter % 21 == 0:
            print(f"END OF MONTH, counter={self.counter}")
            # end of month:
            # firms:
            # use their m_f to pay wages, build buffer and pay profits
            
            self.agents.select(agent_type=Firm).do("pay_wages")
            self.agents.select(agent_type=Firm).do("add_to_buffer", chi=self.chi)
            self.agents.select(agent_type=Firm).do("distribute_profits")


            # households:
            # adjust w_h depending on income
            self.agents.select(agent_type=Household).do("adjust_reservation_wage")




            



