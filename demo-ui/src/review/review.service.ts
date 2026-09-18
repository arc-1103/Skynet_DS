import { Injectable } from "@nestjs/common";
import {
  sampleDriftReport,
  samplePrReview,
  sampleRootCause,
} from "./sample-data";

@Injectable()
export class ReviewService {
  getPrReview() {
    return samplePrReview;
  }

  getDriftReport() {
    return sampleDriftReport;
  }

  getRootCause() {
    return sampleRootCause;
  }
}
