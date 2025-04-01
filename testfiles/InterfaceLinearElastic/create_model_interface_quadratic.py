def generate_abaqus_inp(filename, nodes_per_side=11):

    corner_lines = int(nodes_per_side/2.)
    midpoint_lines = int(nodes_per_side/2.)
    
    with open(filename, "w") as f:
        f.write("** Abaqus Input File: Two Squares with Interface Elements\n")
        
        # Part 1 (Bottom Square)
        f.write("*Part, name=PART1\n")
        f.write("*Node\n")
        node_id = 1
        for j in range(nodes_per_side):
            for i in range(nodes_per_side):
                f.write(f"{node_id}, {i/10.0}, {j/10.0}, 0.0\n")
                node_id += 1
        f.write("*Element, type=CPS8, elset=PART1_ELEM\n")
        elem_id = 1
        for j in range(corner_lines):
            for i in range(corner_lines):
                n1 = 2*j*(2*corner_lines + 1) + 2*i + 1 
                n2 = n1 + 2
                n3 = n1 + 2*nodes_per_side+2
                n4 = n1 + 2*nodes_per_side
                n5 = n1 + 1
                n6 = n1 + nodes_per_side + 2
                n7 = n1 + 2*nodes_per_side + 1
                n8 = n1 + nodes_per_side
                f.write(f"{elem_id}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}\n")
                elem_id += 1                
        f.write("*End Part\n\n")
        
        # Part 2 (Top Square)
        f.write("*Part, name=PART2\n")
        f.write("*Node\n")
        node_id_start = node_id # We pick up where the other part starts
        
        start_node_x2_coord = int(nodes_per_side-1)
        end_node_x2_coord = int(2*nodes_per_side-1)
        node_id = node_id_start
        for j in range(start_node_x2_coord, end_node_x2_coord):
            for i in range(nodes_per_side):
                f.write(f"{node_id}, {i/10.0}, {j/10.0}, 0.0\n")
                node_id += 1
        f.write("*Element, type=CPS8, elset=PART2_ELEM\n")
        for j in range(corner_lines):
            for i in range(corner_lines):
                n1 = 2*j*(2*corner_lines + 1) + 2*i + node_id_start
                n2 = n1 + 2
                n3 = n1 + 2*nodes_per_side+2
                n4 = n1 + 2*nodes_per_side
                n5 = n1 + 1
                n6 = n1 + nodes_per_side + 2
                n7 = n1 + 2*nodes_per_side + 1
                n8 = n1 + nodes_per_side
                f.write(f"{elem_id}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}\n")
                elem_id += 1
        f.write("*End Part\n\n")
        
        # Interface Elements
        f.write("*Part, name=INTERFACE\n")
        f.write("*Node\n")
        # Add interface nodes (mid-edge nodes and corner nodes)
        for i in range(nodes_per_side):
            f.write(f"{200+i}, {i/10.0}, 1.0, 0.0\n")  # Midpoint on the top edge of Part 1 (bottom square)
            f.write(f"{300+i}, {i/10.0}, 1.0, 0.0\n")  # Midpoint on the bottom edge of Part 2 (top square)
        f.write("*Element, type=INTERFACE_ELEM, elset=INTERFACE_SET\n")
        # Interface element connectivity (each interface element connects 8 nodes)
        for i in range(corner_lines):
            # Nodes on the left side (from Part 1 and Part 2)
            left_n1 = 200 + i
            left_n2 = 200 + i + 2
            # Nodes on the right side (from Part 1 and Part 2)
            right_n1 = 300 + i
            right_n2 = 300 + i + 2
            # Midpoint elements (shared nodes between Part 1 and Part 2)
            mid_n1 = 200 + i +1
            mid_n2 = 300 + i +1
            f.write(f"{elem_id}, {left_n1}, {left_n2}, {mid_n1}, {right_n1}, {right_n2}, {mid_n2}\n")
            elem_id += 1
        f.write("*End Part\n\n")
        
        #Define nodal sets for application of dirichlet boundary conditions
        #LEFT
        f.write("*nSet, nSet=left\n")
        for i in range(int(2*nodes_per_side)):
            f.write(str(i*nodes_per_side+1))
            if i < int(2*nodes_per_side-1):
                f.write(",")
            else:
                f.write("\n\n")         
        #Define nodal sets for application of dirichlet boundary conditions
        #RIGHT
        f.write("*nSet, nSet=right\n")        
        for i in range(int(2*nodes_per_side)):
            f.write(str((i+1)*nodes_per_side))
            if i < int(2*nodes_per_side-1):
                f.write(",")
            else:
                f.write("\n\n")
        #Define nodal sets for application of dirichlet boundary conditions
        #TOP
        f.write("*nSet, nSet=top\n")
        top_left_node_number = int((nodes_per_side-1)**2)
        top_right_node_number = int(nodes_per_side**2)
        for i in range(top_left_node_number, top_right_node_number):
            f.write(str(i+1))
            if i < int(top_right_node_number-1):
                f.write(",")
            else:
                f.write("\n\n")
        #Define nodal sets for application of dirichlet boundary conditions
        #BOTTOM
        f.write("*nSet, nSet=bottom\n")               
        bottom_left_node_number = int(0)
        bottom_right_node_number = int(nodes_per_side) 
        for i in range(bottom_right_node_number):
            f.write(str(i+1))
            if i < int(nodes_per_side-1):
                f.write(",")
            else:
                f.write("\n\n")

        
        # Define material
        f.write("*material, name=LinearElastic, id=myMaterial\n")
        f.write("**Isotropic\n")
        f.write("**E   | nu |\n")
        f.write("1.0e4, 0.3\n\n")

        # Define sections
        f.write("*section, name=section1, material=myMaterial, type=solid\n")
        f.write("** element set 'all' is automatically created\n")
        f.write("all\n\n")
        
        # Assembly
        f.write("*Assembly, name=ASSEMBLY\n")
        f.write("*Instance, name=PART1_INST, part=PART1\n")
        f.write("*Instance, name=PART2_INST, part=PART2\n")
        f.write("*Instance, name=INTERFACE_INST, part=INTERFACE\n")
        f.write("*End Assembly\n\n")
        
        # Job
        f.write("*job, name=my_Interface_ElementJob, domain=2d\n\n")

        # select solver
        f.write("*solver, solver=NIST, name=theSolver\n\n")

        # define analysis step
        f.write("*step, solver=theSolver, maxInc=1e0, minInc=1e0, maxNumInc=100, maxIter=25, stepLength=1\n")
        f.write("** The first step: set the BCs\n")
        f.write("dirichlet, name=1, nSet=left,       field=displacement, 2=0\n")
        f.write("dirichlet, name=2, nSet=right,       field=displacement, 2=0\n")        
        f.write("dirichlet, name=3, nSet=top, field=displacement, 1=0,2=0\n")
        f.write("dirichlet, name=4, nSet=bottom, field=displacement, 1=0,2=0\n\n")

        f.write("*step, solver=theSolver, maxInc=1e0, minInc=1e0, maxNumInc=10, maxIter=25, stepLength=1\n")
        f.write("** The second step: apply the incremental displacement\n")
        f.write("dirichlet, name=3, nSet=top, field=displacement, 1=0,2=0.1\n")
        f.write("dirichlet, name=4, nSet=bottom, field=displacement, 1=0,2=-0.1\n\n")
        #f.write("distributedload, name=dload, surface=surfaceRight, type=pressure, magnitude=0.15, f(t)='t**2'\n")

        # define output
        f.write("*fieldOutput\n")
        f.write("** The results we are interested in")
        f.write("create=perElement, name=strain, elSet=all, result=strain, quadraturePoint=0:8, f(x)='np.mean(x,axis=1)'\n")
        f.write("create=perElement, name=stress, elSet=all, result=stress, quadraturePoint=0:8, f(x)='np.mean(x,axis=1)'\n")
        f.write("create=perNode, name=displacement, elSet=all, field=displacement, result=U\n\n")

        f.write("*output, type=ensight, name=myEnsightOutput\n")
        f.write("** For visualization in Paraview etc.\n")
        f.write("create=perNode, fieldOutput=displacement\n")
        f.write("create=perElement, fieldOutput=stress\n")
        f.write("create=perElement, fieldOutput=strain\n")
        f.write("configuration, overwrite=yes\n\n")

        f.write("*output, type=monitor, name=myMonitor\n")
        f.write("** Directly print to the terminal\n")
        f.write("fieldOutput=RF_left\n\n")
        
#        # Boundary Conditions
#        f.write("*Boundary\n")
#        f.write("PART1_BOTTOM, 1, 1, -0.1\n")
#        f.write("PART1_BOTTOM, 2, 2, 0\n")
#        f.write("PART2_TOP, 1, 1, 0.1\n")
#        f.write("PART2_TOP, 2, 2, 0\n")
#        f.write("PART1_LEFT, 2, 2, 0\n")
#        f.write("PART2_LEFT, 2, 2, 0\n")
#        f.write("PART1_RIGHT, 2, 2, 0\n")
#        f.write("PART2_RIGHT, 2, 2, 0\n")
    
# Generate the Abaqus input file
generate_abaqus_inp("interface_model_quadratic.inp")

