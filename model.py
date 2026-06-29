import numpy as np
import mesa
import random
from agents import Household, Firm

class LegnickModel(mesa.Model):
    """the Legnick model"""
    
    def __init__(self, n_households = 1000, n_firms = 100, seed=333, alpha = 0.9, n = 7, ld = 3, chi = 0.1, gamma = 24, delta = 0.019):
        super().__init__(rng=seed)
        self.n_households = n_households
        self.n_firms = n_firms
        self.counter = 0
        self.alpha = alpha
        self.n = n
        self.ld = ld
        self.chi = chi
        self.gamma = gamma
        self.delta = delta

        # create agents
        Household.create_agents(model=self, n=n_households)
        Firm.create_agents(model=self, n=n_firms)

        Households = self.agents.select(agent_type = Household)
        firms = list(self.agents.select(agent_type = Firm))

        # initialise trading connectons (type a)

        for h in Households:
            
            type_a = random.sample(firms, 7)
            h.type_a_connections = type_a

    def step(self):
        self.counter += 1
        if self.counter % 21 == 1:
            # beginning of month
            # firms:
            # each decide how to set w_f
            self.agents.select(agent_type =Firm).do("set_wages", gamma=self.gamma, delta=self.delta)
            # each decide p_f if i_f unsatisfactory level
            # households:
            # each search for better type_a connections
            # if type_b_connection = None, go to random f to search for open position
            # decide how much m_h to spend on consumption goods
            self.agents.select(agent_type=Household).do("monthly_consumption", alpha=self.alpha)

        # daily: 
        # households:
        # use their m_h to buy goods from random type_a connection
        # demand is equally spread thru month
        
        self.agents.select(agent_type=Household).do("buy_goods", n=self.n)
        
        # firms produce
        self.agents.select(agent_type=Firm).do("produce", ld=self.ld)


        if self.counter % 21 == 0:
            # end of month:
            # firms:
            # use their m_f to pay wages, build buffer and pay profits
            
            self.agents.select(agent_type=Firm).do("pay_wages")
            self.agents.select(agent_type=Firm).do("add_to_buffer", chi=self.chi)
            self.agents.select(agent_type=Firm).do("distribute_profits")


            # households:
            # adjust w_h depending on income
            self.agents.select(agent_type=Household).do("adjust_reservation_wage")



            



