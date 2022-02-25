import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp

def SIRSyst(t, z, k, tau, N):
    S, I, R = z
    return [-k*S*I, k*S*I - 1/tau*I, 1/tau*I]

def NetworkSetup(CellPopD, Type='ER', pER=0.5, WSneig=10, pWS=0.5, PAneig=10, pPA=0.5):
    '''
    A function to initialise a graph in terms of agents with certain attributes. Returns an 
    object of the agent network.
    
    CellPopD: dictionary of 2-entry lists. Key -value pairs of cells' ID with list of corresponding 
    population and danger.
    
    Type: Any of the NetworkX of prototype network implementations of the following: ER-->Erdos-Renyi graph, 
    WS-->Watts-Strogatz, PA-->Power-law cluster graph, BA-->Barabasi-Albert graph.
    
    Returns: The agents as a Networkx object.
    '''
    AgNo = sum( [ i[0] for i in CellPopD.values() ] )  # No of agents in the simulation
    # Initialisation
    if Type == 'ER':  # complete graph (all to all)
        AG = nx.erdos_renyi_graph(AgNo, pER)
    elif Type == 'WS':  # original Watts-Strogatz small-world network
        AG = nx.watts_strogatz_graph(AgNo, WSneig, pWS)
    elif Type == 'PA':  # power-law cluster graph of Holme and Kim with preferential attachment
        AG = nx.powerlaw_cluster_graph(AgNo, PAneig, pPA)
    elif Type == 'BA':  # Barabasi-Albert random preferential attachment
        AG = nx.barabasi_albert_graph(AgNo, PAneig)
    else:
        raise NameError("Unrecognised network type.")
    
    first, temporig, tempAgDanger, tempCellDanger, tempCellHDI = 0, {}, {}, {}, {}
    for i in CellPopD:
        last = first + CellPopD[i][0]
        Estimate = np.zeros( ( CellPopD[i][0], len(CellPopD) ) )  # No of agents X No of cells for estimate
        for cell in enumerate(CellPopD):  # an estimate of danger is generated for every cell
            if CellPopD[cell[1]][1] > 0:
                AgDanger = np.random.triangular(0, CellPopD[cell[1]][1]/2, CellPopD[cell[1]][1], size=CellPopD[i][0])  # initialisation of the danger perceptions per cell from a triangular distribution with mode half the true danger in the corresponding cell
                AgDanger = np.round(AgDanger, 4)
                Estimate[:,cell[0]] = AgDanger
            else:
                AgDanger = np.zeros(CellPopD[i][0])  # no alarm at start for cells where no attack has taken place
                Estimate[:,cell[0]] = AgDanger
        tempAgDanger.update( { j[1] : Estimate[j[0]] for j in enumerate( range(first, last) ) } )
        tempCellDanger.update( { j : CellPopD[i][1] for j in range(first, last) } )
        temporig.update( { j : i for j in range(first, last) } )
#         tempCellHDI.update( { j : CellPopD[i][2] for j in range(first, last) } )  # HDI of cell-country attribute
        first += CellPopD[i][0]
    nx.set_node_attributes(AG, tempAgDanger, 'AgDanger')
    nx.set_node_attributes(AG, tempCellDanger, 'CellDanger')
    nx.set_node_attributes(AG, tempCellDanger, 'CellDangerStart')
    nx.set_node_attributes(AG, temporig, 'orig')
#     nx.set_node_attributes(AG, tempCellHDI, 'HDI')
    return AG
    
def SpatialDangerProgression(AgNtw, t, Tnulls, progression='LinDec', T=100, k=1.5, tau=5, N=100, tol=1e-5):
    '''
    Function for progressing the true danger in each cell of an agent given an initial value (at t=0).
    Requires the running time as well as the time at which the danger will have fully abated (or have fallen 
    below a given tolerance if the progression's decay is asymptotic). In case the SIR model is selected 
    all the intrinsic parameters need to be specified (presets available for a given time range).
    
    LinDec: If the initial value of danger assumes any of the extreme ones then under the linear model they remain 
    there.
    
    ExpDec: If the initial value of danger assumes any of the extreme ones then under the exponential model they 
    remain there.
    
    SIR: The danger follows the infectious (normalised) infectious class of the SIR model.
    '''
    tempSt = np.array( list( nx.get_node_attributes(AgNtw, 'CellDangerStart').values() ) )  # the initial cells' danger
    temp = np.array( list( nx.get_node_attributes(AgNtw, 'CellDanger').values() ) ).astype(float)  # the present cells' danger
        
    for Tn in Tnulls:
        idx = (temp > 0) & (tempSt == Tnulls[Tn][1])  # since danger is monotonous the initial non 0, 1 values of each cell should coincide for temp and tempSt
        if (progression == 'LinDec'):            
            LinC = tempSt[idx]/Tnulls[Tn][0]  # determining the linear coefficients needed to annul the cell's danger at the specified time (Tn[0])
            if np.any(temp < tol):  # once the danger has disappeared assume it remains so under linear decay assumptions
                temp[temp < tol] = 0.
            if np.any(temp > 0):
                temp[idx] = np.round( tempSt[idx] - LinC*t, 4 )  # linear decay
            aux = { i[0] : i[1] for i in enumerate(temp) }
        elif (progression == 'ExpDec'):
            ExpC = ( np.log(tempSt[idx]) - np.log(tol) )/Tnulls[Tn][0]  # determining the exponential decay's coefficient needed to reach a given tolerance at Tnull
            if np.any(ExpC) < 0:
                raise ValueError('Negative exponent. Impossible exponential decay. Check the relation between the initial danger and the tolerance.')
            if np.any(temp < tol):
                temp[temp < tol] = 0  # vanquish the danger below the given tolerance
            if np.any(temp > tol):
                temp[idx] = np.round( tempSt[idx] * np.exp(-ExpC*t), 4 )  # exponential decay
            aux = { i[0] : i[1] for i in enumerate(temp) }
        elif (progression == 'SIR'):  # infectious class of SIR as danger case
            if np.any(temp < tol):
                temp[temp < tol] = 0  # vanquish the danger below the given tolerance
            if np.any(temp > tol):
                sol = solve_ivp(SIRSyst, [1, T], [1-Tnulls[Tn][1], Tnulls[Tn][1], 0], args=(k, tau, N), dense_output=True)  # we always assume that the R class is 0 at t=0
                tm = np.linspace(1, T, 3*T)
                z = sol.sol(tm)
                inf = z[1][:-1:3]
                temp[idx] = np.round( inf[t-1] * np.ones_like(temp[idx]), 4 )  # the infection-danger at the specified time (note the index-time step relation as per the simulation setup. This is due to compatibility with the design of the network of danger as a scalar instead of an array)
            aux = { i[0] : i[1] for i in enumerate(temp) }
        else:
            raise NameError('Unrecognised danger progression')
    nx.set_node_attributes(AgNtw, aux, 'CellDanger')
    
def Influence(AgNtw, Model='Deffuant', epsilon=0.2, mu=0.5):
    '''
    Function to calculate the influence on the perception term according to the Deffuant (selective bias), 
    F-J (assimilative) and J-A (repulsive) model with network considerations on a time step.
    Returns a vector with the updated influences for all the agents. Passing the distance matrix as an 
    argument had to be a compensation of run-time to the expense of memory and efficient function calling.
    For networks up to 10**3 nodes this has been tested and is inconsequential. For large networks it 
    should play a role and might be something worth taking into account for design.
    '''
    AgDim, CellNo = len(AgNtw.nodes), len(AgNtw.nodes[1]['AgDanger'])
    InfMat = np.zeros( (AgDim, CellNo) )
    aux1 = np.array( list( nx.get_node_attributes(AgNtw, 'AgDanger').values() ) )
    for i in AgNtw.nodes():
        aux3 = aux1 - AgNtw.nodes[i]['AgDanger']  # the common difference term for all opinion models
        if Model == 'Deffuant':  # Deffuant model (selective biased influence)
            aux2 = AgNtw.nodes[i]['AgDanger'] - aux1
            idaux = (aux2 <= epsilon) & (-epsilon <= aux2)  # epsilon test for selective influence
            aux3[idaux] = mu * aux3[idaux]  # selective bias for influence
            aux3[~idaux] = 0  # the rest are unaffected
            InfMat[i] = AgNtw.nodes[0]['DistMat'][i] @ aux3/AgDim  # Deffuant evolution term
        elif Model == 'F-J':  # Friedkin-Johnsen model implementation (assimilative influence)
            weight = np.random.rand(AgDim)  # assign randomly weights among the agents
            weight = weight/weight.sum()  # normalise
            InfMat[i] = mu * AgNtw.nodes[0]['DistMat'][i] @ (weight[i] * aux3/AgDim)  # F-J evolution term with smoothening and weights
        elif Model == 'J-A':  # Jaeger-Amblard model (repulsive influence)
            aux2 = AgNtw.nodes[i]['AgDanger'] - aux1
            idaux = (aux2 <= epsilon) & (-epsilon <= aux2)  # epsilon test for selective influence
            aux3[idaux] = mu * aux3[idaux]  # J-A attractive influence
            aux3[~idaux] = mu * ( np.ones_like(aux3[~idaux]) - 2*np.abs(aux3[~idaux]) )  # J-A repulsive influence
            InfMat[i] = AgNtw.nodes[0]['DistMat'][i] @ aux3/AgDim  # J-A evolution term            
        else:
            raise NameError('Unrecognised influence model.')
    InfMat = np.round_( aux1 + InfMat, 4 ) # evolution
    temp = { j[1] : InfMat[j[0]] for j in enumerate( range(AgDim) ) }
    nx.set_node_attributes(AgNtw, temp, 'Influence' )
    
def Estimation(AgNtw, CellPopD, Method='Triang'):
    '''
    Function to calculate the personal estimation of each agent for their cell and all others, given a true
    level of danger of their cell (local information). Possible methods to perform this estimation are based 
    on diverse distributions, parametrised by the true level of danger. These distribution are the triangular 
    (Triang), the Gaussian (Gauss), the Laplacian (Laplace) and a power law (PwrL).
    '''
    first, tempAgDanger = 0, {}
    for i in CellPopD:
        last = first + CellPopD[i][0]
        Estimate = np.zeros( ( CellPopD[i][0], len(CellPopD) ) )  # No of agents X No of cells for estimate
        for cell in enumerate(CellPopD):  # an estimate of danger is made for every cell
            if Method == 'Triang':  # Triangular distribution with the true danger of the cell as the mode of the agent's estimation for all cells
                AgDanger = np.random.triangular(0, AgNtw.nodes[first]['CellDanger'], 1, size=CellPopD[i][0])  # first suffices for all the agents in the same cell because they all see the same true danger
            elif Method == 'Gauss':  # Gaussian distribution with the true danger of the cell as the location parameter of the agent's estimation for all cells with the scale parameter as the maximum possible fluctuation
                ScPar = 1
                AgDanger = np.abs( np.random.normal(AgNtw.nodes[first]['CellDanger'], ScPar, size=CellPopD[i][0]) )
                AgDanger[AgDanger > 1] = np.random.choice( [0,1], size=len(AgDanger[AgDanger > 1]) )  # estimates off the scale parameter (std) are randomly assigned to either end of the perceptions' spectrum
            elif Method == 'Laplace':  # as in the case of the Gaussian but for a Laplace distribution
                ScPar = 1
                AgDanger = np.abs( np.random.laplace(AgNtw.nodes[first]['CellDanger'], ScPar, size=CellPopD[i][0]) )
                AgDanger[AgDanger > 1] = np.random.choice( [0,1], size=len(AgDanger[AgDanger > 1]) )  # similar logic to the Gauss
            elif Method == 'PwrL':  # a power law distribution with the true danger being the exponent of the numpy power law routine
                if AgNtw.nodes[first]['CellDanger'] == 0.:  # the numpy routine for a power law with a=0 is not defined, hence we identify a power-law bias of the zero true danger as coinciding with zero
                    AgDanger = np.zeros(CellPopD[i][0])
                else:
                    AgDanger = np.random.power(AgNtw.nodes[first]['CellDanger'], size=CellPopD[i][0])
            else:
                raise NameError('Unrecognised estimation method.')
            AgDanger = np.round(AgDanger, 4)
            Estimate[:,cell[0]] = AgDanger
        tempAgDanger.update( { j[1] : Estimate[j[0]] for j in enumerate( range(first, last) ) } )
        first += CellPopD[i][0]
    nx.set_node_attributes(AgNtw, tempAgDanger, 'Estimation')
    
def AggregationMeasure(AgNtw, Method='Mean'):
    '''
    A function to coestimate (aggregate) the personal measurement-assessment of danger for each agent, 
    for their own and for every other cell. The aggregate measure used is the mean (Mean).
    '''
    AgEstimates = nx.get_node_attributes(AgNtw, 'Estimation')
    AgInfluences = nx.get_node_attributes(AgNtw, 'Influence')
    shp = len( np.unique( list( nx.get_node_attributes(AgNtw, 'orig').values() ) ) )
    temp = {}
    for i in AgNtw.nodes:
        conc = np.concatenate( (AgEstimates[i].reshape(1,shp), AgInfluences[i].reshape(1,shp) ), axis=0 )
        if Method == 'Mean':
            temp.update( { i : np.round( np.mean( conc, axis=0 ), 4 ) } )
        else:
            raise NameError('Unrecognised aggregation method.')
    nx.set_node_attributes(AgNtw, temp, 'AgDanger')
