package run;

import java.io.File;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import org.deckfour.xes.classification.XEventClass;
import org.deckfour.xes.classification.XEventClassifier;
import org.deckfour.xes.classification.XEventNameClassifier;
import org.deckfour.xes.info.XLogInfo;
import org.deckfour.xes.info.XLogInfoFactory;
import org.deckfour.xes.model.XLog;
import org.deckfour.xes.in.XesXmlGZIPParser;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.graphbased.directed.petrinet.elements.Place;
import org.processmining.models.graphbased.directed.petrinet.elements.Transition;
import org.processmining.models.semantics.petrinet.Marking;

import au.unimelb.evaluation.AlignmentSetup;
import au.qut.apromore.importer.ImportPetriNet;
import org.processmining.plugins.petrinet.replayresult.PNRepResult;
import org.processmining.plugins.connectionfactories.logpetrinet.TransEvClassMapping;

import org.processmining.antialignments.ilp.antialignment.AntiAlignmentPlugin;
import org.processmining.antialignments.ilp.antialignment.AntiAlignmentParameters;
import org.processmining.antialignments.ilp.antialignment.AntiAlignmentValues;
import org.processmining.antialignments.ilp.util.AntiAlignments;
import org.processmining.antialignments.ilp.antialignment.HeuristicAntiAlignmentAlgorithm;

/**
 * M4 runner: uses AutomataConformance's PNML import + AlignmentSetup,
 * then calls AntiAlignmentPlugin.basicCodeStructureWithAlignments() directly.
 * No ProM GUI framework needed.
 */
public class M4Runner {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: M4Runner <log.xes.gz> <model.pnml>");
            System.exit(1);
        }
        String logPath = args[0];
        String modelPath = args[1];

        System.err.println("=== M4 Anti-Alignment (no GUI) ===");
        System.err.println("Log:   " + logPath);
        System.err.println("Model: " + modelPath);

        // 1. Load XES
        System.err.print("Loading log... ");
        XesXmlGZIPParser parser = new XesXmlGZIPParser();
        Collection<XLog> logs = parser.parse(new File(logPath));
        XLog log = logs.iterator().next();
        System.err.println("done (" + log.size() + " traces)");

        // 2. Load PNML (via ImportPetriNet which uses FakePluginContext)
        System.err.print("Loading model... ");
        Object[] pnmlResult = ImportPetriNet.importPetriNetAndMarking(modelPath);
        Petrinet net = (Petrinet) pnmlResult[0];
        Marking initMarking = (Marking) pnmlResult[1];
        Marking finalMarking = null;
        // ImportPetriNet doesn't return final marking; try to find it
        for (Object o : pnmlResult) {
            if (o instanceof Marking && o != initMarking) {
                finalMarking = (Marking) o;
                break;
            }
        }
        if (finalMarking == null) finalMarking = new Marking();
        int nTrans = net.getTransitions().size();
        int nPlaces = net.getPlaces().size();
        System.err.println("done (" + nTrans + "t, " + nPlaces + "p)");

        // 3. Alignments (no GUI via AlignmentSetup)
        System.err.print("Computing alignments... ");
        AlignmentSetup alignSetup = new AlignmentSetup(net, log);
        PNRepResult alignments = alignSetup.getAlignment(null, initMarking, finalMarking, false, 0);
        System.err.println("done (" + alignments.size() + " alignments)");

        // 4. Build TransEvClassMapping manually
        System.err.print("Building mapping... ");
        XEventClassifier classifier = new XEventNameClassifier();
        TransEvClassMapping mapping = new TransEvClassMapping(classifier, 
            new XEventClass("DUMMY", -1));
        for (Transition t : net.getTransitions()) {
            if (t.isInvisible() || t.getLabel() == null || t.getLabel().isEmpty()) {
                mapping.put(t, null);
            } else {
                mapping.put(t, new XEventClass(t.getLabel(), 
                    t.getLabel().hashCode()));
            }
        }
        System.err.println("done (" + mapping.size() + " entries)");

        // 5. Compute anti-alignments & generalization
        System.err.print("Computing anti-alignments... ");
        long t0 = System.currentTimeMillis();
        
        HeuristicAntiAlignmentAlgorithm algorithm = new HeuristicAntiAlignmentAlgorithm(
            net, initMarking, finalMarking, log, alignments, mapping);
        AntiAlignmentParameters params = new AntiAlignmentParameters(5, 1.0, 2, 2.0);
        AntiAlignments aa = algorithm.computeAntiAlignments(null, params);
        AntiAlignmentValues values = algorithm.computePrecisionAndGeneralization(aa);
        
        long elapsed = System.currentTimeMillis() - t0;
        double gen = values.getGeneralization();
        double prec = values.getPrecision();
        
        System.err.println("done (" + elapsed + "ms)");
        System.out.println("\n=== RESULTS ===");
        System.out.println("Precision:     " + prec);
        System.out.println("Generalization: " + gen);
        System.out.println("Time:          " + elapsed + "ms");
        System.out.println("\nCSV: " + logPath + "," + modelPath + "," 
            + (elapsed / 1000.0) + "," + gen);
    }
}
