from datetime import datetime
import PyANGConsole as cs
import PyANGKernel as gk
import csv
import AAPI as aapi


model = gk.GKSystem.getSystem().getActiveModel()
# global edge_detector_dict
# edge_detector_dict = {}


now = datetime.now()
westbound_section = [506, 563, 24660, 568, 462]
eastbound_section = [338, 400, 461, 24650, 450]
sections = [22208, 568, 22211, 400]
node_id = 3344

interval = 3*60

replication_name = aapi.ANGConnGetReplicationId()
reps = [8050297, 8050315, 8050322]
for repli in reps:
    replication = model.getCatalog().find(repli)
current_time = now.strftime('%d-%m-%Y-%H-%M:%S')

'''with open('keith_{}.csv'.format(replication_name), 'w') as csvFile:
    data = []
    fieldnames = ['time', 'flow', 'occupancy', 'queue', 'stop_time', 'approach_delay']
    csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
    csv_writer.writeheader()'''


def get_delay_time(section_id):
    west = []
    east = []
    for section_id in westbound_section:
        estad_w = aapi.AKIEstGetGlobalStatisticsSection(section_id, 0)
        if estad_w.report == 0:
            print('Delay time: {} - {}'.format(section_id, estad_w.DTa))
        west.append(estad_w.DTa)

    for section_id in eastbound_section:
        estad_e = aapi.AKIEstGetGlobalStatisticsSection(section_id, 0)
        if estad_e.report == 0:
            print('Delay time: {} - {}'.format(section_id, estad_e.DTa))
        east.append(estad_e.DTa)

    west_ave = sum(west)/len(west)
    east_ave = sum(east)/len(east)

    print("Average Delay Time: WestBound {}".format(west_ave))
    print("Average Delay Time: EastBound {}".format(east_ave))


def sum_queue(section_id):
    catalog = model.getCatalog()
    node = catalog.find(node_id)
    in_edges = node.getEntranceSections()

    section_list = [edge.getId() for edge in in_edges]

    for section_id in section_list:
        section = catalog.find(section_id)
        num_lanes = section.getNbLanesAtPos(section.length2D())
        queue = sum(aapi.AKIEstGetCurrentStatisticsSectionLane(
            section_id, i, 0).LongQueueAvg for i in range(num_lanes))

        queue = queue * 5 / section.length2D()

    print('SUM QUEUE {} : {}'.format(node_id, total_queue))


def AAPILoad():
    return 0


def AAPIInit():
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    # print( "AAPIManage" )
    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):

    #fieldnames = ['time', 'flow','occupancy', 'queue','stop_time','approach_delay']
    # print( "AAPIPostManage" )
    detectors = [508309, 508325, 508324, 508312, 508310, 508322, 508315, 508311]
    if time % (900) == 0:
        if replication_name == 8050297:
            filename = "Data_%i.csv" % replication_name
            Results_head = open(filename, 'w')
            Results_head.write('flow, occupancy, queue, stop_time, approach_delay\n')
            for section_id in sections:
                estad = aapi.AKIEstGetGlobalStatisticsSection(section_id, 0)
                if (estad.report == 0):
                    queue = estad.LongQueueAvg
                    stop_time = estad.STa
                    time = time
                    flow = estad.Flow
                    approach_delay = aapi.AKIEstGetPartialStatisticsNodeApproachDelay(node_id)
                    for det in detectors:
                        occ_list = []
                        occ_list.append(aapi.AKIDetGetTimeOccupedAggregatedbyId(det, 0)/100)
                    occupancy = max(occ_list)
                    Results_row = open(filename, 'a')
                    Results_row.write("%i,%4f,%4f,%4f,%4f\n" %
                                      (flow, occupancy, queue, stop_time, approach_delay))
                    Results_row.close()

                #print(time, queue, flow, occupancy, stop_time, approach_delay)

                '''if replication_name == 8050297:
                    with open('{}.csv'.format(replication_name), 'a') as csvFile:
                        csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                        csv_writer.writerow({'time': time, 'flow': flow, 'occupancy': occupancy,
                                             'queue': queue, 'stop_time': stop_time, 'approach_delay': approach_delay})

                elif replication_name == 8050315:
                    with open('{}.csv'.format(replication_name), 'a') as csvFile:
                        csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                        csv_writer.writerow({'time': time, 'flow': flow, 'occupancy': occupancy,
                                             'queue': queue, 'stop_time': stop_time, 'approach_delay': approach_delay})

                elif replication_name == 8050322:
                    with open('{}.csv'.format(replication_name), 'a') as csvFile:
                        csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                        csv_writer.writerow({'time': time, 'flow': flow, 'occupancy': occupancy,
                                             'queue': queue, 'stop_time': stop_time, 'approach_delay': approach_delay})'''

    return 0


def AAPIFinish():
    # print("AAPIFinish")
    return 0


def AAPIUnLoad():
    return 0


def AAPIPreRouteChoiceCalculation(time, timeSta):
    return 0


def AAPIEnterVehicle(idveh, idsection):
    return 0


def AAPIExitVehicle(idveh, idsection):
    return 0


def AAPIEnterPedestrian(idPedestrian, originCentroid):
    return 0


def AAPIExitPedestrian(idPedestrian, destinationCentroid):
    return 0


def AAPIEnterVehicleSection(idveh, idsection, atime):
    return 0


def AAPIExitVehicleSection(idveh, idsection, atime):
    return 0
