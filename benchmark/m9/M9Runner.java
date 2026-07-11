package m9;
import org.deckfour.xes.model.XLog;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.semantics.petrinet.Marking;
import au.unimelb.negativeEventsClasses.PetrinetLogMapper;
import au.unimelb.evaluation.NegativeEventsTask;
import au.qut.apromore.importer.ImportProcessModel;
import au.qut.apromore.importer.ImportEventLog;

public class M9Runner {
    public static void main(String[] args) throws Exception {
        String logPath = args[0], modelPath = args[1], label = args[2];
        String cfg = args.length > 3 ? args[3] : "cobefra";
        long t0 = System.currentTimeMillis();
        Object[] pm = new ImportProcessModel().importPetriNetAndMarking(modelPath);
        Petrinet net = (Petrinet) pm[0];
        Marking marking = (Marking) pm[1];
        XLog log = new ImportEventLog().importEventLog(logPath);
        PetrinetLogMapper mapper = PetrinetLogMapper.getStandardMap(log, net);
        double gen;
        if (cfg.equals("m8")) {
            gen = NegativeEventsTask.getMetricValue(log, net, marking, mapper, 1,1,true,true,true,20,20,true,false,true,false, "generalization");
        } else {
            gen = NegativeEventsTask.getMetricValue(log, net, marking, mapper, 0,0,true,false,false,-1,-1,true,true,true,false, "generalization");
        }
        long ms = System.currentTimeMillis() - t0;
        System.out.println("M9_RESULT\t" + label + "\tcfg=" + cfg + "\tgen=" + gen + "\truntime_ms=" + ms);
    }
}
