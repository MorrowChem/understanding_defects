import networkx as nx
from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from ovito.io import import_file, export_file
from ovito.modifiers import CreateBondsModifier, \
    CoordinationAnalysisModifier,ComputePropertyModifier, \
    ExpressionSelectionModifier, DeleteSelectedModifier
from ovito.pipeline import ModifierInterface
from traits.api import Dict

# TODO: differentiate search medium from isomorphic medium?

def read_write_defects(infile, outfile, readbonds=False, delete_bulk=True):
    """
    Reads a structure file and adds topology information based on a cutoff,
    then writes the modified data to the LAMMPS bond style file.

    Args:
        infile (str): The path to the input file.
        outfile (str): The path to the output file.
        readbonds (bool or str, optional): Determines whether to read bonds or not. If set to False, 
            coordination analysis and bond creation will be performed. If set to 'c_cn', only bond creation 
            will be performed. If set to any other string, molecule identification will be performed. 
            Defaults to False.

    Returns:
        None
    """
    
    pipeline = import_file(infile)
    if not readbonds:
        pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=2.85))
        pipeline.modifiers.append(CreateBondsModifier(cutoff=2.85))
        pipeline.modifiers.append(ComputePropertyModifier(output_property='c_cn',
                                                      expressions='Coordination'))
    
    elif readbonds=='c_cn':
        pipeline.modifiers.append(CreateBondsModifier(cutoff=2.85))
    else:
        pipeline.modifiers.append(ComputePropertyModifier(output_property='c_cn',
                                                      expressions='MoleculeIdentifier'))
        

    pipeline.modifiers.append(ComputePropertyModifier(output_property='del',
                                                      expressions='abs(c_cn-4)'))
    pipeline.modifiers.append(ExpressionSelectionModifier(expression='del == 0'))

    if delete_bulk:
        pipeline.modifiers.append(DeleteSelectedModifier())
    data = pipeline.compute()
    
    export_file(data, outfile,
                format='xyz',
                columns=['Particle Identifier', 'Particle Type', 'Position.X', 'Position.Y', 'Position.Z', 'c_cn'])
    
    export_file(data, outfile[:-4]+'_bond.data',
                format='lammps/data',
                atom_style='bond')
     
    return

def recursive_delete(infile, outfile, bulk_level=3, extra_keys=[], recalc_cn=True, **extra_props):
    """
    Recursively determines the neighborhoods of defects in a structure file and deletes the rest of the 'bulk' structure.
    
    Args:
        infile (str): The path to the input structure file with defects identified.
        outfile (str): The path to the output file where the modified structure will be saved.
        bulk_level (int, optional): The bonded range after which atoms become 'bulk' and are marked
        for deletion. Defaults to 3.
        **extra_props: Additional properties to be added to the output file.
    
    Returns:
        None
    """
    
    pipeline = import_file(infile)
    
    pipeline.modifiers.append(AddArrayModifier(extra_props=dict(extra_props)))
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=2.85))
    pipeline.modifiers.append(CreateBondsModifier(cutoff=2.85))
    if recalc_cn:
        pipeline.modifiers.append(ComputePropertyModifier(output_property='c_cn',
                                                        expressions='Coordination'))

    pipeline.modifiers.append(ComputePropertyModifier(output_property='del0',
                                                      expressions='abs(c_cn-4)'))

    pipeline.modifiers.append(ComputePropertyModifier(output_property='del1',
                                                      expressions='del0',
                                                      neighbor_expressions='c_cn != 4')) # last one redundant and therefore misleading. test without
    for i in range(1,bulk_level+1):
        pipeline.modifiers.append(ComputePropertyModifier(output_property=f'del{i+1}',
                                                        expressions=f'del{i}',
                                                        neighbor_expressions=f'del{i} != 0'))
    
    pipeline.modifiers.append(ExpressionSelectionModifier(expression=f'del{bulk_level} == 0'))
    pipeline.modifiers.append(DeleteSelectedModifier())
    
    data = pipeline.compute()
    export_file(data, outfile,
                format='xyz',
                columns=['Particle Identifier', 'Particle Type', 
                         'Position.X', 'Position.Y', 'Position.Z', 
                         'c_cn'] + [key for key in extra_props.keys()] + \
                            extra_keys + \
                            [f'del{i}' for i in range(1,bulk_level+1)])
    
    export_file(data, outfile[:-4]+'_bond.data',
                format='lammps/data',
                atom_style='bond')

class DefectGraph():
    
    def __init__(self, file, cn_name='c_cn', es_name='c_pe_at', 
                 extra_keys=[], medium=[], max_outer=4, bulk_level=3,
                 max_sep=0, defects=[2,3,5,6], skip_structure_prep=False,
                 include_bulk_in_iso=False, recalc_cn=True):
        """
        Initialise object to store defect environments as graphs based on bonding topology.

        Parameters:
        - file (str): Path to the structure file.
        - cn_name (str, optional): Name for coordination number data. Defaults to 'c_cn'.
        - es_name (str, optional): Name for local energy data. Defaults to 'c_pe_at'.
        - extra_keys (list, optional): Additional keys from structure file to keep in memory.
        - medium (list, optional): Atom types to be considered non-defective (medium like 'aether').
            For silicon, this is 14, 24, 34, which denote 4', 4'', 4''' neighboutd
        - max_outer (int, optional): Maximum outer value. Defaults to 4.
        - bulk_level (int, optional): keep non-defective atoms up to this bonding range from defect centres
        - max_sep (int, optional): Maximum separation value between defect centres to consider as a single graph. Defaults to 0.
        - defects (list, optional): List of defect coordination number labels. Defaults to [2, 3, 5, 6] for 4-fold silicon.
        - skip_structure_prep (bool, optional): Skip structure preparation if already done.
        - include_bulk_in_iso (bool, optional): Include bulk/medium atoms in isomorphism test if True.
        - recalc_cn (bool, optional): Recalculate coordination numbers using cutoff radius and overwrite c_cn in structure file.

        Notes:
        - The 'set_types' method is used internally to set atom types based on coordination numbers and bulk levels.
        """
        
        if not skip_structure_prep:
            print('Preparing structure files...', end='')
            recursive_delete(file, 'tmp/tmp_def.xyz',
                            bulk_level=bulk_level, extra_keys=[es_name]+extra_keys, recalc_cn=recalc_cn)
        at_def = read('tmp/tmp_def.xyz')
        self.at = read('tmp/tmp_def_bond.data',
                format='lammps-data', style='bond')
        self.at.arrays[cn_name] = at_def.arrays['c_cn']
        print(f" done. Structure has {len(self.at)} atoms")
        try:
            self.at.arrays[es_name] = at_def.arrays[es_name]
        except:
            print('No local energies found')
            self.at.arrays[es_name] = np.zeros(len(self.at))

        print('setting types...', end='')
        self.set_types(at_def, bulk_level=bulk_level)
        print(' done')
        del at_def

        self.ts = self.types
        self.cell = self.at.cell
        
        self.bs = self.at.arrays['bonds']
        self.bsr = make_bs_reciprocal(self.bs)
        self.cn_name = cn_name
        self.es_name = es_name
        self.es = self.at.arrays[es_name]
        self.positions = self.at.positions
        
        self.extra_keys = extra_keys
        self.medium = medium
        self.max_outer = max_outer
        self.max_sep = max_sep
        self.defects = defects
        self.include_bulk_in_iso = include_bulk_in_iso
        
        self.get_defect_graphs(self.at, cn_name=cn_name, es_name=es_name,
                               extra_keys=extra_keys, medium=medium,
                                max_outer=max_outer, max_sep=max_sep, defects=defects)
        
    def set_types(self, at_def, bulk_level=3):
        # overridable to declare your own type-setting system. We base it on coordination number for aSi
        types = at_def.arrays['c_cn'].astype(np.int64) # use coordination numbers to identify types
        for i in range(1, bulk_level+1):
            print(at_def.arrays[f'del{i}'])
            types[
                np.argwhere(np.all(np.vstack(
                            (at_def.arrays[f'del{i}'] != 0, types == 4)),
                            axis=0)).squeeze()
                            ] = i*10+4 # 14 denotes 4', 24 - 4'', etc.
        at_def.set_atomic_numbers(types)
        at_def.arrays['type'] = types
        self.at.set_atomic_numbers(types)
        self.at.arrays['type'] = types
        self.types = types
        print("Type summary: ", np.unique(types, return_counts=True))
            
    def get_defect_graphs(self, at, cn_name='c_cn', es_name='c_pe_at',
                          extra_keys=[], medium=[], max_outer=2,
                        max_sep=0, defects=[2,3,5,6]):
    
        edge_list = []
        for i, val in enumerate(self.bs):
            if val != '_' and self.types[i] in defects:
                for j in val.split(','):
                    cn = int(j[:-3])
                    if self.types[cn] in defects:
                        edge_list.append(tuple([i, cn]))
                    
        node_list = []
        for i in np.arange(len(self.bs)):
            if self.types[i] in defects:
                node_list.append((i, {"N": self.types[i], 'e':self.es[i], "positions": self.positions[i],         
                            **{j:at.arrays[j][i] for j in extra_keys}}))

        self.G = nx.Graph()
        self.G.add_nodes_from(node_list)
        self.G.add_edges_from(edge_list)
        self.G.graph['positions'] = at.arrays['positions']

        Gcon = list(nx.connected_components(self.G))
        print('Found connected components: ', len(Gcon))

        clus = [len(i) for i in Gcon]
        clus_5 = [len(i) for i in Gcon if np.all(
                        [self.G.subgraph(i).nodes[j]['N'] == 5.0 for j in self.G.subgraph(i).nodes]
                        )]
        self.clus = clus
        self.clus5 = clus_5
        
        # find only defect subgraphs
        S = [self.G.subgraph(c).copy() for c in nx.connected_components(self.G)] # include isolated defects # if len(c)>1]
        
        print('Found {} subgraphs'.format(len(S)))
        self.S = S
        tmp_Gs = []
        if medium != []:
            print('Medium is ', medium)
            group_energies = False
            
            print('Searching recursively for neighbouring medium')
            print('Linking nodes with max separation ', max_sep)
            link_counter = 0
            for i in range(len(S)):
                
                # reconstruct the graph up to level max_outer (need to go recursively)
                visited = []
                flag = True
                search_counter = 1
                med_level = 0
                while flag: # iterate over nodes until there are no more edges to add
                    if search_counter > max_outer:
                        break
                    flag = False
                    med = medium[med_level:]
                    med_level += 1
                    search_counter += 1
                    
                    for node in tuple(S[i].nodes):
                        if node in visited:
                            continue
                        visited.append(node)
                        
                        edge = (self.bs[node].strip('_') + ',' + self.bsr[node].strip('_')).strip(',')
                        if edge != '':
                            for j in edge.split(','):
                                cn = int(j[:-3])
                                if self.types[cn] in med:
                                    ind = int(j[:-3])
                                    S[i].add_edge(node, ind)
                                    S[i].add_node(ind, **{"N": self.types[ind], 'e':self.es[ind], 'positions':self.positions[ind],
                                                            **{l:at.arrays[l][ind] for l in extra_keys}})
                                    flag = True  # indicates that we're still growing after looping over all unvisited nodes
                                if self.types[cn] in defects and max_sep > 0:  # find defects linked through medium
                                    ind = int(j[:-3])
                                    S[i].add_edge(node, ind)
                                    S[i].add_node(ind, **{"N": self.types[ind]+10, 'e':self.es[ind], 'positions':self.positions[ind],
                                                **{l:at.arrays[l][ind] for l in extra_keys}})
                                    flag = True
                
                self.label_S_es(S[i], extra_keys)

                nodes_to_include = [j for j in S[i].nodes if S[i].nodes[j]['N'] not in medium]
                if max_sep > 0:
                    for ct, j in enumerate(nodes_to_include[:-1]):
                        for k in nodes_to_include[ct+1:]:
                            sp = nx.shortest_path(S[i], j, k)
                            if len(sp) > 2 and len(sp) <= 2+max_sep and \
                                len(np.intersect1d([S[i].nodes[n]['N'] for n in sp[1:-1]],
                                                            medium)):
                                link_counter += 1
                                nodes_to_include.append(sp[1])
                                    
                tmp_Gs.append(S[i].subgraph(nodes_to_include))
            
        else:
            print('No medium selected')
            group_energies = True
            for i in range(len(S)):
                self.label_S_es(S[i], extra_keys)
            tmp_Gs = S
            
        self.med_Gs = tmp_Gs
        
        if medium != []:
            print('found {} extra links due to non-zero max separation'.format(link_counter))
        print('Length of tmp_Gs', len(tmp_Gs))
        Sg_t_cts, Sg_t, Sg_t_inds = self.test_isomorphism(extra_keys=extra_keys,
                                        cn_name=cn_name, es_name=es_name, group_energies=group_energies,
                                        include_bulk_in_iso=self.include_bulk_in_iso)
        
        self.Sg_t = Sg_t
        self.Sg_t_inds = Sg_t_inds
        self.Sg_t_cts = np.array(Sg_t_cts)
        self.S = S
        
        print('Summary: \n')
        for i in range(len(Sg_t)):
            if i < 10:
                print('{:50}     {}'.format(str([Sg_t[i].nodes[k]['N'] for k in Sg_t[i].nodes if Sg_t[i].nodes[k]['N'] not in self.medium]), Sg_t_cts[i]))
            if i==11:
                print('...')
            if i>len(Sg_t)-2:
                print('{:50}     {}'.format(str([Sg_t[i].nodes[k]['N'] for k in Sg_t[i].nodes if Sg_t[i].nodes[k]['N'] not in self.medium]), Sg_t_cts[i]))
        return

    def condense_defect_subgraphs(self, medium=[4], extra_keys=[], max_sep=0, **kwargs):
        """
        Condenses all isomorphic examples of defect environments and 
        collects the information (e.g. energies) into a single representative graph.

        Parameters:
            S (list): A list of graphs representing defect environments.
            medium (list, optional): Node type to ignore when deciding isomorphism. Defaults to an empty list.
            extra_keys (list, optional): Additional keys for node attributes. Defaults to an empty list.
            max_sep (int, optional): Maximum separation between defect nodes to allow inclusion in a single subgraph. Defaults to 0.

        Returns:
            tuple: A tuple containing the counts of each unique graph and a list of the unique graphs.
        """

        tmp_Gs = []
        S = self.S
        
        for ct, G in enumerate(S):
            
            nodes_to_include = [i for i in G.nodes if G.nodes[i]['N'] not in medium]
            if max_sep > 0:
                for n in G.nodes:
                    for ct, i in enumerate(nodes_to_include):
                        for j in nodes_to_include[ct+1:]:
                            sp = nx.shortest_path(G, i, j)
                            if len(sp) == max_sep+2:
                                nodes_to_include.append(sp[1])

            tmp_G = G.subgraph(nodes_to_include)
            
            tmp_G.graph['es_cn'] = {}
            for k in extra_keys:
                tmp_G.graph[k+'_cn'] = {}
            for c in G.nodes:
                cn = G.nodes[c]['N']
                if cn not in tmp_G.graph['es_cn'].keys():
                    tmp_G.graph['es_cn'][cn] = []
                    for k in extra_keys:
                        tmp_G.graph[k+'_cn'][cn] = []

                tmp_G.graph['es_cn'][cn].append(G.nodes[c]['e'])
                for k in extra_keys:
                        tmp_G.graph[k+'_cn'][cn].append(G.nodes[c][k])
                        
            tmp_Gs.append(tmp_G.copy())
    
        G_cts, G = self.test_isomorphism(extra_keys=extra_keys, **kwargs)
        
        return G_cts, G
        
    def test_isomorphism(self, cn_name='c_cn', es_name='c_pe_at',
                         extra_keys=[], group_energies=True, include_bulk_in_iso=False):
        """
        Test whether defect environments are isomorphic from their bonding graph.
        It ignores the 'medium': the 4-fold coordinated structure that decorates the defect environment.

        Parameters:
        - S (list): A list of bonding graphs representing defect environments.
        - cn_name (str): The name of the attribute that stores the coordination numbers in the bonding graph.
        - es_name (str): The name of the attribute that stores the total energies in the bonding graph.
        - extra_keys (list): A list of additional attribute names to consider in the bonding graph.
        - group_energies (bool): Flag indicating whether to group the total energies or not.

        Returns:
        - Sg_t_cts (ndarray): An array of counts for each isomorphism group.
        - Sg_t (list): A list of bonding graphs representing the isomorphism groups.
        - Sg_t_inds (list): A list of indices indicating which defect environments belong to each isomorphism group.
        """
        
        S = self.S
        Sg_t = []
        Sg_t_cts = []
        Sg_t_inds = []
        
        if group_energies:
            stack = np.vstack
        else:
            # not possible to simply stack the total energies if ignoring some medium (lengths vary)
            stack = np.hstack
        
        print('Categorising {} subgraphs'.format(len(S)))
        for i, val in enumerate(S):
            flag = True
            for j, val2 in enumerate(Sg_t):
                if not include_bulk_in_iso:
                    def_val = val.subgraph([k for k in val.nodes \
                                            if val.nodes[k]['N'] not in self.medium])
                    def_val2 = val2.subgraph([k for k in val2.nodes \
                                              if val2.nodes[k]['N'] not in self.medium])
                    test = j != i and nx.is_isomorphic(def_val, def_val2,
                                                       node_match=lambda x,y: x['N'] == y['N'])
                else:
                    test = j != i and nx.is_isomorphic(val, val2,
                                                       node_match=lambda x,y: x['N'] == y['N'])
                if test:
                    flag = False
                    Sg_t_cts[j] += 1
                    Sg_t_inds[j].append(i)
                    Sg_t[j].graph['es'] = stack((Sg_t[j].graph['es'], val.graph['es']))

                    Sg_t[j].graph['coms'] = np.vstack((Sg_t[j].graph['coms'],
                                                       val.graph['positions'].mean(axis=0)))
                    Sg_t[j].graph['es_av'].append(val.graph['es'].mean())
                    for k in extra_keys:
                        if k != 'positions':
                            Sg_t[j].graph[k] = stack((Sg_t[j].graph[k], val.graph[k]))
                            Sg_t[j].graph[k+'_av'].append(val.graph[k].mean())

                    for c in val.graph['es_cn'].keys():
                        
                        try:
                            Sg_t[j].graph['es_cn'][c].extend(val.graph['es_cn'][c])
                            for k in extra_keys:
                                Sg_t[j].graph[k+'_cn'][c].extend(val.graph[k+'_cn'][c])
                        except:
                            print('found no {} at subgraph {}'.format(c, j))
                            Sg_t[j].graph['es_cn'][c] = val.graph['es_cn'][c]
                            for k in extra_keys:
                                Sg_t[j].graph[k+'_cn'][c] = val.graph[k+'_cn'][c]
                        
                    break

            if flag:
                Sg_t.append(val.copy())
                Sg_t[-1].graph['coms'] = Sg_t[-1].graph['positions']
                Sg_t[-1].graph['es_av'] = [Sg_t[-1].graph['es'].mean()]
                Sg_t_inds.append([i])
                for k in extra_keys:
                    if k != 'positions':
                        Sg_t[-1].graph[k+'_av'] = [Sg_t[-1].graph[k].mean()]

                Sg_t_cts.append(1)
                
        for sg in range(len(Sg_t)):
            Sg_t[sg].graph['es_av'] = np.array(Sg_t[sg].graph['es_av'])
            for k in extra_keys:
                if k != 'positions':
                    Sg_t[sg].graph[k+'_av'] = np.array(Sg_t[sg].graph[k+'_av'])
            for c in Sg_t[sg].graph['es_cn'].keys():
                Sg_t[sg].graph['es_cn'][c] = np.array(Sg_t[sg].graph['es_cn'][c])
                for k in extra_keys:
                    if k != 'positions':
                        Sg_t[sg].graph[k+'_cn'][c] = np.array(Sg_t[sg].graph[k+'_cn'][c])   
        Sg_t_cts = np.array([i for i in Sg_t_cts if i!=0])
        Sg_t = [x for _, x in sorted(zip(Sg_t_cts, Sg_t),
                                     key=lambda pair: pair[0], reverse=True)]
        Sg_t_inds = [x for _, x in sorted(zip(Sg_t_cts, Sg_t_inds),
                                          key=lambda pair: pair[0], reverse=True)]
        Sg_t_cts = sorted(Sg_t_cts, reverse=True)
        
        return Sg_t_cts, Sg_t, Sg_t_inds

    def write_graphs_to_file(self, file, type_map=None):

        ats = []; ats2 = []
        for s in self.Sg_t:
            pos = []
            n = []
            for nod in s.nodes:
                pos.append(s.nodes[nod]['positions'])
                n.append(s.nodes[nod]["N"])

            if type_map is not None:
                n = [type_map[i] for i in n]
            pos = np.array(pos)
            at = Atoms(positions=pos, numbers=n, cell=self.cell, pbc=True)
            ats2.append(at.copy())
            at.set_positions(at.get_positions() - at.positions[0] + self.cell[0,0]/2)
            at.wrap()
            at.cell=None
            at.set_pbc((False, False, False))
            ats.append(at)

        write(file, ats)
        
        return ats

    def label_S_es(self, Si, extra_keys=[]):
    
        Si.graph['es'] = np.array([Si.nodes[j]['e'] for j in Si.nodes])
        for k in extra_keys:
            Si.graph[k] = np.array([Si.nodes[j][k] for j in Si.nodes])

        Si.graph['es_cn'] = {}
        for k in extra_keys:
            Si.graph[k+'_cn'] = {}
        for c in Si.nodes:
            cn = Si.nodes[c]['N']
            if cn not in Si.graph['es_cn'].keys():
                Si.graph['es_cn'][cn] = []
                for k in extra_keys:
                    Si.graph[k+'_cn'][cn] = []

            Si.graph['es_cn'][cn].append(Si.nodes[c]['e'])
            for k in extra_keys:
                    Si.graph[k+'_cn'][cn].append(Si.nodes[c][k])

    def draw(self, G):
        """
        TODO: fix this nicely
        Draw the bonding graph of a defect environment.

        Parameters:
            G (nx.Graph): The bonding graph to draw.

        Returns:
            None
        """
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, font_weight='bold')
        plt.show()


class AddArrayModifier(ModifierInterface):
    
    extra_props = Dict({})
    
    def modify(self, data, **kwargs):
        
        for key, array in self.extra_props.items():
            print(key, array.shape)
            data.particles_.create_property(key, data=array)
            

def make_bs_reciprocal(bs):
    """
    Convert the bonding description from LAMMPS bond to a two-way reciprocal representation.

    Parameters:
        bs (numpy.ndarray): The bonding description in LAMMPS bond format.

    Returns:
        numpy.ndarray: The two-way reciprocal representation of the bonding description.
    """
    bsr = np.empty(bs.shape, dtype='U100')
    bsr[:] = ''

    for i, val in enumerate(bs):
        if val != '_':
            for j in val.split(','):
                tmp_str = ',' + str(i) + '(1)'
                bsr[int(j[:-3])] = bsr[int(j[:-3])] + tmp_str

    for i, val in enumerate(bsr):
        if val == '':
            bsr[i] = '_'
        else:
            bsr[i] = val[1:]

    return bsr