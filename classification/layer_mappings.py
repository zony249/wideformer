
#teacher_depth --> student_depth --> list of teacher layers
LAYER_MAPPING = {
    12: {
        3: [],
        6: [], 
        9: [],
        12: []
    },  
    24: { 
        3: []
    }, 
    28: {
        3: [9, 18, 27],
        7: [],
        14: [], 
        28: []
    }
}


#teacher_depth --> student_depth --> list of teacher layers
LAYER_COPYING = {
    28: {
        3: [0, 14, 27], 
    } 
}