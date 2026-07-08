import numpy as np
import mesa

class Household(mesa.Agent):
    """Household agents"""

    def __init__(self, model):
        # pass the params to parent class
        super().__init__(model)

        self.w_h = 3.0 # reservation wage
        self.m_h = 3100 # liquidity
        self.type_a_connections = [] # list of firms 
        self.type_b_connection = None # employment
        self.c_r_h = 0 # demand
        self.income = 0 #income from last month
        self.demand_const = False # was this h demand const?
        self.demand_const_shops = {}

    def monthly_consumption(self, alpha):
        """
        find each households monthly consumption need. 
        - find avg of all trade connection prices
        - use eq 12 to find consumption
        """
        prices = 0
        for a in self.type_a_connections:
            prices += a.p_f
        p_i_h = prices / len(self.type_a_connections)
        self.c_r_h = min((self.m_h / p_i_h)**alpha, self.m_h / p_i_h)
    
    def buy_goods(self, n):
        """
        each household buys goods from a random type a connection
        - if they cant satisfy the demand - move on to another random type a connection
        - if hh cant afford - buy all they can afford 
        - stop when demand 95% satisfied
        """
        demand = self.c_r_h / 21
        og_demand = demand
        shops = self.random.sample(self.type_a_connections, len(self.type_a_connections))
        for shop in shops[:n]:
            # track demand for each firm
            
            if shop.i_f >= demand and self.m_h >= (shop.p_f * demand):
                self.m_h -= shop.p_f * demand
                # guard from going -ve 
                self.m_h = max(self.m_h, 0.0)
                shop.m_f += shop.p_f * demand
                shop.i_f -= demand
                shop.demand += demand
                demand = 0

            # what if they both cant afford it.... will it just try and do this?
            elif self.m_h < (shop.p_f * demand):
                demand_new = self.m_h / shop.p_f
                # do i need to then check here if the firm can do it 
                if shop.i_f >= demand_new:
                    self.m_h -= shop.p_f * demand_new
                    # stop from going -ve from fp drift 
                    self.m_h = max(self.m_h, 0.0)
                    shop.m_f += shop.p_f * demand_new
                    shop.i_f -= demand_new
                    shop.demand += demand_new
                    demand -= demand_new
            elif shop.i_f < demand:  
                self.demand_const = True  
                self.m_h -= shop.p_f * shop.i_f
                # guard from going -ve 
                self.m_h = max(self.m_h, 0.0)
                shop.m_f += shop.p_f * shop.i_f
                self.demand_const_shops[shop] = self.demand_const_shops.get(shop, 0) + (demand - shop.i_f)
                demand -= shop.i_f
                shop.demand += shop.i_f
                shop.i_f = 0
            if demand <= 0.05 * og_demand:
                break   

    def adjust_reservation_wage(self):
        """ adjust reservation wage according to last months income"""
        if self.type_b_connection is not None:
            if self.income > self.w_h:
                self.w_h = self.income
        else:
            self.w_h = self.w_h * 0.9

    def search_connections(self, psi_price, xi, psi_quant):
        if self.random.random() < psi_price:
            typeA = self.random.choice(self.type_a_connections)
            all_firms = set(self.model.agents.select(agent_type=Firm))
            no_type_as = list(all_firms - set(self.type_a_connections))
            weights = [len(f.workers) for f in no_type_as]
            if sum(weights) == 0:
                new_firm = self.random.choice(no_type_as)
            else:
                new_firm = self.random.choices(no_type_as, weights=weights, k=1)[0]

            if new_firm.p_f < (1 - xi) * typeA.p_f:
                self.type_a_connections.remove(typeA)
                self.type_a_connections.append(new_firm)
            
        if self.demand_const == True:
            if self.random.random() < psi_quant:
                
                shops = list(self.demand_const_shops.keys())
                weights = list(self.demand_const_shops.values())
                shop = self.random.choices(shops, weights=weights, k=1)[0]
                if shop in self.type_a_connections:

                    all_firms = set(self.model.agents.select(agent_type=Firm))
                    no_type_as = list(all_firms - set(self.type_a_connections))
                    self.type_a_connections.remove(shop)
                    new_shop = self.random.choice(no_type_as)
                    self.type_a_connections.append(new_shop)
        
        self.demand_const = False
        self.demand_const_shops = {}

    def job_search(self, beta, pie):
        # unemployed
        if self.type_b_connection == None:
            firms = self.random.sample(list(self.model.agents.select(agent_type=Firm)), beta)
            for f in firms:
                if f.open_position == True:
                    if f.w_f >= self.w_h: # "greater than his currently received wage"? or meant to be w_h
                        self.type_b_connection = f
                        f.workers.append(self)
                        f.open_position = False
                        break
        # satisfied
        else:
            if self.type_b_connection.w_f >= self.w_h:
                if self.random.random() < pie:
                    all_firms = set(self.model.agents.select(agent_type=Firm))
                    no_type_b = list(all_firms - {self.type_b_connection})
                    f = self.random.choice(no_type_b)
                    if f.open_position == True:
                        if f.w_f >= self.type_b_connection.w_f:
                            self.type_b_connection.workers.remove(self)
                            self.type_b_connection = f
                            f.workers.append(self)
                            f.open_position = False
            # unsatisfied
            else: 
                all_firms = set(self.model.agents.select(agent_type=Firm))
                no_type_b = list(all_firms - {self.type_b_connection})
                f = self.random.choice(no_type_b)                
                if f.open_position == True:
                    if f.w_f >= self.w_h:
                        self.type_b_connection.workers.remove(self)
                        self.type_b_connection = f
                        f.workers.append(self)
                        f.open_position = False


class Firm(mesa.Agent):
    """Firm agents"""

    def __init__(self, model):
        super().__init__(model)

        self.m_f = 0.0 # liquidity
        self.i_f = 0.0 # inventory
        self.p_f = 25 # goods price
        self.w_f = 1428 # wage rate
        self.workers = [] # list of workers
        self.buffer = 0
        self.open_position = False #open position boolean so there can be only one 
        #self.n_positions = 10 - dont need this anymore
        self.months_full = 0
        self.demand = 0
        self.to_fire = [] # workers that are being fired next month

    def produce(self, ld):
        """produce goods"""
        l_f = len(self.workers)
        self.i_f += ld * l_f

    def pay_wages(self):
        """
        firm pays wages to all workers
        either at set wage rate or if cant afford, drop wage
        """
        # add buffer to liquidity and then reset to 0
        self.m_f += self.buffer
        self.buffer = 0
        #guard for if 0 work=ers in firm
        if len(self.workers) == 0:
            return
        l_f = len(self.workers)
        if self.m_f >= self.w_f * l_f:
            for h in self.workers:
                self.m_f -= self.w_f
                h.m_h += self.w_f
                # store income
                h.income = self.w_f
        else:
            new_wage = self.m_f / l_f
            for h in self.workers:
                self.m_f -= new_wage
                h.m_h += new_wage
                # store income
                h.income = new_wage

    def add_to_buffer(self, chi):
        """add left over liquidity to buffer """
        l_f = len(self.workers)
        if self.m_f > chi * self.w_f * l_f:
            # add to buffer - full amt
            self.buffer += chi * self.w_f * l_f
            self.m_f -= chi * self.w_f * l_f
        elif self.m_f > 0:
            # add to buffer - not full amt
            self.buffer += self.m_f
            self.m_f = 0

    def distribute_profits(self):
        """distribute remaining profit to all hh according to their liquidity"""
        if self.m_f > 0: 
            # distribute profit for each house proportional to m_h
            households = self.model.agents.select(agent_type=Household)
            total_liquid = sum(h.m_h for h in households)
            for h in households:
                h.m_h += (self.m_f / total_liquid) * h.m_h
        self.m_f = 0

    def set_wages(self, gamma, delta):
        """
        set wages for the month
        """
        # draw from uniform dist
        mu = self.random.uniform(0, delta)
        l_f = len(self.workers)
        if self.open_position == True:
            self.w_f = self.w_f * (1 + mu)
            self.months_full = 0

        elif self.months_full > gamma:
            self.w_f = self.w_f * (1 - mu)
            self.months_full = 0
        else:
            self.months_full += 1

    def set_employment(self, phi_emp_upper, phi_emp_lower):
        """
        set employment rate dependant on inventory levels
        """
        i_f_upperbar = phi_emp_upper * self.demand
        i_f_lowerbar = phi_emp_lower * self.demand
        # only if inventory under i_f_lowbar and all positions are currently full
        if self.i_f < i_f_lowerbar:
            self.open_position = True
        elif self.i_f > i_f_upperbar and len(self.workers) > 0:
            to_fire = self.random.choice(self.workers)
            self.to_fire.append(to_fire)
            self.open_position = False


    def fire_workers(self):
        """ fire workers from last month - 1 month delay"""
        
        for h in self.to_fire:
            if h in self.workers:
                self.workers.remove(h)
                h.type_b_connection = None           
        self.to_fire = []

    def set_prices(self, phi_price_upper, phi_price_lower, ld, theta, phi_emp_upper, vartheta, phi_emp_lower):
        # only consider if inventory in the right amount
        i_f_upperbar = phi_emp_upper * self.demand
        i_f_lowerbar = phi_emp_lower * self.demand
        mc_f = self.w_f / (21 * ld) # marginal costs for the month rather than the day?
        v = self.random.uniform(0, vartheta)

        p_f_upperbar = phi_price_upper * mc_f
        p_f_lowerbar = phi_price_lower * mc_f

        if self.i_f > i_f_upperbar:
            # if the price is above p_f_upperbar 
            if self.p_f > p_f_lowerbar:
                # lower prices with prob theta 
                if self.random.random() < theta:
                    self.p_f = self.p_f * (1 - v)
        
        if self.i_f < i_f_lowerbar:
            # if the price is lower that p_f_lowerbar
            if self.p_f < p_f_upperbar:
                # up prices with prob theta 
                if self.random.random() < theta:
                    self.p_f = self.p_f * (1 + v)

        self.demand = 0
        




