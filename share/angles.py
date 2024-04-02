import ovito
from ovito.modifiers import CreateBondsModifier, CoordinationAnalysisModifier, \
                            ComputePropertyModifier, BondAnalysisModifier
from ovito.io import import_file
import numpy as np


#TODO document and tidy up these. Just getting it working for now

def calc_badf_select(pipeline, cutoff1=2.85, bins=360):
    
    # Should delete existing badf from pipeline before adding this one (otherwise need
    #  to regen the pipe)

    # could delete non-involved atoms at this stage
    # # or change the particle types to give some more information (e.g. s1-s1-s2, s1-s1-s1 etc.)
    # cb = CreateBondsModifier(cutoff=cutoff1)
            # mode=CreateBondsModifier.Mode.Pairwise)
    # cb.bond_type.id = 1
    
    # pipeline.modifiers.append(cb)
    print(pipeline.modifiers)
    pipeline.modifiers.append(BondAnalysisModifier(
        partition=BondAnalysisModifier.Partition.ByParticleType, bins=bins))

    data = pipeline.compute()

    return data


# amalgamate some similar angles together using convention
def amalgamate_badf_centres(hist):
    new_components = {}
    done = []

    for i, val in enumerate(hist.y.component_names):
        nname = val.split('-')[1]
        if not nname in done:
            new_components[nname] = np.array(hist.y.array[:, i])
            done.append(nname)
        elif nname in new_components.keys():
            new_components[nname] += np.array(hist.y.array[:, i])
                    
    return new_components
    
    

def label_by_top_sep(file, nneigh=3, cutoff1=2.85, defect=5, 
                        extra_conditions='', extra_neigh_conditions='', extra_counter_neigh_conditions='',
                        soap_data=None, label=None):

    pipeline = import_file(file)
    def insert_soap_data_modifier(frame, data):
        for ct, d in enumerate(soap_data):
            data.particles_.create_property(label[ct], data=d)

    if soap_data is not None:
        pipeline.modifiers.append(insert_soap_data_modifier)

    cb = CreateBondsModifier(cutoff=cutoff1)
    cb.bond_type.id = 1
    pipeline.modifiers.append(cb)
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff1))

    pipeline.modifiers.append(ComputePropertyModifier(output_property='bond_type',
                                                    expressions=['@1.Coordination',
                                                                '@2.Coordination',
                                                                'BondType'],
                                                    operate_on='bonds'))

    expressions = 'Coordination == {}'.format(defect)
    if extra_conditions != '':
        expressions += ' && {}'.format(extra_conditions)
    pipeline.modifiers.append(ComputePropertyModifier(output_property='s1',  # label selected defects that also obey extra conditions
                                                    expressions=[expressions],
                                                    neighbor_expressions=[extra_neigh_conditions],
                                                    operate_on='particles'))
    
    counter_expressions = 'Coordination != 4 && s1 <= 0'
    if extra_conditions != '':
        counter_expressions += ' && ( Coordination != {} || Coordination == {}'.format(defect, defect) +\
                                                         ' && ({})? 0: 1'.format(extra_conditions) + ')'



    pipeline.modifiers.append(ComputePropertyModifier(output_property='s_1',  # label not-considered defects
                                                    expressions=[counter_expressions],
                                                    neighbor_expressions=[extra_counter_neigh_conditions],
                                                    operate_on='particles'))
    

    pipeline.modifiers.append(ComputePropertyModifier(output_property='s2',  # label immediate neighbours of defects that are non-defective
                                                    expressions=['(Coordination == 4) ? 0 : -100'],
                                                    neighbor_expressions=[expressions],
                                                    operate_on='particles',
                                                    cutoff_radius=cutoff1))

    pipeline.modifiers.append(ComputePropertyModifier(output_property='s2_5',  # label immediate neighbours of not-considered defects that are non-defective
                                                    expressions=['s2 ? 0 : -100'],
                                                    neighbor_expressions=[f'(Coordination == {defect})'],
                                                    operate_on='particles',
                                                    cutoff_radius=cutoff1))                                                

    pipeline.modifiers.append(ComputePropertyModifier(output_property='s_2',  # label immediate neighbours of defects that are non-defective
                                                    expressions=['(Coordination == 4) ? 0 : -100'],
                                                    neighbor_expressions=[counter_expressions],
                                                    operate_on='particles',
                                                    cutoff_radius=cutoff1))
    print('exploring neighbours:')
    for i in range(2, nneigh):
        print(i+1)                                                 
        pipeline.modifiers.append(ComputePropertyModifier(output_property=f's{i+1}',
                                                    expressions=[f's{i}'],
                                                    neighbor_expressions=[f's{i} > 0'],
                                                    operate_on='particles',
                                                    cutoff_radius=cutoff1))

        pipeline.modifiers.append(ComputePropertyModifier(output_property=f's_{i+1}',
                                                    expressions=[f's_{i}'],
                                                    neighbor_expressions=[f's_{i} > 0'],
                                                    operate_on='particles',
                                                    cutoff_radius=cutoff1))
    pts = None
    
    def change_types(frame, data):
        
        for i in range(2, nneigh+1):  # for purposes of easy argmax later, set everything to 1 or 0 if negative

            data.particles_[f's{i}_'][data.particles[f's{i}']>=1] = 1
            data.particles_[f's{i}_'][data.particles[f's{i}']<1] = 0
        
            data.particles_[f's_{i}_'][data.particles[f's_{i}']>=1] = 1
            data.particles_[f's_{i}_'][data.particles[f's_{i}']<1] = 0
            
        nonlocal pts 
        pts = [np.any(np.array([data.particles[f's_{i}'] for i in range(1, nneigh+1)]), axis=0)] + \
                        [data.particles[f's{i}'] for i in range(1, nneigh+1)]
        print(np.max(pts), np.min(pts))
        
        #  now set 'bulk' Si as type 1, non-select defects as type 2, defects as 3, and subsequent shells as 4, 5 etc.
        new_type = np.argmax(pts, axis=0)+2 - np.invert(np.any(pts, axis=0))
        print(tmp := np.unique(new_type, return_counts=True), sum(tmp[1]))
        print((sum(tmp[1][1:])))

        ##### NEED TO ASSIGN ATOMS THEIR TYPE IN AN ARRAY ####
        data.particles_.particle_types_ =  new_type
        names = ['d{}'.format(defect)] + [str(i) for i in range(1,30)]
        
        data.particles_.particle_types_.types.append(
                ovito.data.ParticleType(id=2, name='d', 
                        color=(0.0, 1.0, 1.0)))
        
        for i in range(nneigh+1):
            data.particles_.particle_types_.types.append(
                ovito.data.ParticleType(id=3+i, name='Si{}'.format(names[i]),
                        color=(0.0, 1.0, 0.0+i/nneigh)))


    pipeline.modifiers.append(change_types)

    data = pipeline.compute()

    return data, pipeline, pts